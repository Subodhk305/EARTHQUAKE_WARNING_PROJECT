"""
IRIS FDSN Web Service Waveform Fetcher.

Uses ObsPy FDSN client to:
  1. Find the nearest seismic stations to a given lat/lon.
  2. Download 1-hour mini-SEED waveform data (Z, N, E channels).
  3. Pre-process: detrend, bandpass filter, resample.
  4. Return numpy array suitable for CNN-LSTM input.
"""
from __future__ import annotations

import io
import logging
import asyncio
import numpy as np
from functools import partial
from typing import Optional

import aiohttp
from obspy import read, Stream, UTCDateTime
from obspy.clients.fdsn import Client as FDSNClient
from obspy.signal.filter import bandpass

# Fix for scipy.signal.hann deprecation
try:
    from scipy.signal.windows import hann
except ImportError:
    from scipy.signal import hann

from earthquake_service.config import settings
from earthquake_service.utils.cache import cache_get, cache_set

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
SAMPLE_RATE = 100          # Hz — resample target
WINDOW_SECONDS = 600       # 10-minute waveform window
N_CHANNELS = 3             # Z, N, E
FREQMIN = 1.0              # bandpass low  (Hz)
FREQMAX = 40.0             # bandpass high (Hz)
MAX_STATIONS = 5           # limit nearby stations for speed
CACHE_TTL = 300            # 5 minutes


# ── Helpers ───────────────────────────────────────────────────────────────────

def _normalize(arr: np.ndarray) -> np.ndarray:
    """Zero-mean, unit-variance normalization per channel."""
    mean = arr.mean(axis=-1, keepdims=True)
    std = arr.std(axis=-1, keepdims=True) + 1e-8
    return (arr - mean) / std


def _stream_to_array(st: Stream, target_samples: int) -> np.ndarray:
    """
    Convert ObsPy Stream to (N_CHANNELS, target_samples) numpy array.
    Channels are ordered Z, N/1, E/2.  Missing channels are zero-filled.
    """
    channel_map = {"Z": 0, "N": 1, "1": 1, "E": 2, "2": 2}
    result = np.zeros((N_CHANNELS, target_samples), dtype=np.float32)
    for tr in st:
        ch = tr.stats.channel[-1].upper()
        idx = channel_map.get(ch)
        if idx is None:
            continue
        data = tr.data[:target_samples].astype(np.float32)
        pad = target_samples - len(data)
        if pad > 0:
            data = np.pad(data, (0, pad))
        result[idx] = data
    return _normalize(result)


def _fetch_waveform_sync(latitude: float, longitude: float) -> np.ndarray:
    """Blocking IRIS fetch — run in executor to avoid blocking event loop."""
    client = FDSNClient("IRIS")
    t_end = UTCDateTime()
    t_start = t_end - WINDOW_SECONDS

    # ── Find nearest stations ──────────────────────────────────────────────
    try:
        inventory = client.get_stations(
            latitude=latitude,
            longitude=longitude,
            maxradius=5.0,          # degrees ~550 km
            channel="HH?,BH?",      # broadband + high-gain
            level="channel",
            starttime=t_start,
            endtime=t_end,
        )
    except Exception as e:
        logger.warning(f"Station inventory fetch failed: {e}")
        return np.zeros((N_CHANNELS, SAMPLE_RATE * WINDOW_SECONDS), dtype=np.float32)

    stations = []
    for net in inventory:
        for sta in net:
            stations.append((net.code, sta.code))
            if len(stations) >= MAX_STATIONS:
                break
        if len(stations) >= MAX_STATIONS:
            break

    if not stations:
        logger.warning("No IRIS stations found near (%.3f, %.3f)", latitude, longitude)
        return np.zeros((N_CHANNELS, SAMPLE_RATE * WINDOW_SECONDS), dtype=np.float32)

    # ── Download waveforms ─────────────────────────────────────────────────
    bulk = []
    for net, sta in stations:
        bulk.append((net, sta, "*", "HH?,BH?", t_start, t_end))

    try:
        st = client.get_waveforms_bulk(bulk)
    except Exception as exc:
        logger.warning("Bulk waveform fetch failed (%s). Trying station-by-station.", exc)
        st = Stream()
        for net, sta in stations[:2]:
            try:
                st += client.get_waveforms(net, sta, "*", "HH?,BH?", t_start, t_end)
            except Exception as e:
                logger.debug(f"Station {net}.{sta} failed: {e}")
                continue

    if not st:
        logger.warning("Empty waveform stream for (%.3f, %.3f)", latitude, longitude)
        return np.zeros((N_CHANNELS, SAMPLE_RATE * WINDOW_SECONDS), dtype=np.float32)

    # ── Pre-process ────────────────────────────────────────────────────────
    try:
        st.detrend("demean")
        st.detrend("linear")
        st.taper(0.05)
        st.filter("bandpass", freqmin=FREQMIN, freqmax=FREQMAX, corners=4, zerophase=True)
        st.resample(SAMPLE_RATE)
        st.merge(method=1, fill_value="interpolate")
    except Exception as e:
        logger.error(f"Error processing waveform: {e}")
        return np.zeros((N_CHANNELS, SAMPLE_RATE * WINDOW_SECONDS), dtype=np.float32)

    target_samples = SAMPLE_RATE * WINDOW_SECONDS
    return _stream_to_array(st, target_samples)


# ── Public async API ─────────────────────────────────────────────────────────

async def fetch_waveform(latitude: float, longitude: float) -> np.ndarray:
    """
    Fetch and preprocess seismic waveform data from IRIS.
    Results are cached in Redis for CACHE_TTL seconds.

    Returns:
        np.ndarray of shape (N_CHANNELS, time_steps) — float32
    """
    cache_key = f"waveform:{latitude:.3f}:{longitude:.3f}"
    cached = await cache_get(cache_key)
    if cached:
        logger.debug("Waveform cache HIT for key %s", cache_key)
        return np.array(cached["data"], dtype=np.float32)

    logger.info("Fetching IRIS waveform for (%.4f, %.4f)...", latitude, longitude)
    loop = asyncio.get_event_loop()
    try:
        array = await asyncio.wait_for(
            loop.run_in_executor(
                None,
                partial(_fetch_waveform_sync, latitude, longitude),
            ),
            timeout=30.0,
        )
    except asyncio.TimeoutError:
        logger.error("IRIS waveform fetch timed out — returning zeros.")
        array = np.zeros((N_CHANNELS, SAMPLE_RATE * WINDOW_SECONDS), dtype=np.float32)
    except Exception as exc:
        logger.error("IRIS fetch error: %s — returning zeros.", exc)
        array = np.zeros((N_CHANNELS, SAMPLE_RATE * WINDOW_SECONDS), dtype=np.float32)

    # Cache as nested list (JSON-serialisable)
    await cache_set(cache_key, {"data": array.tolist()}, ttl=CACHE_TTL)
    return array


async def get_recent_seismicity(
    latitude: float,
    longitude: float,
    radius_km: float = 200.0,
    days: int = 30,
) -> list[dict]:
    """
    Query USGS FDSN for recent earthquake events near the given location.
    Returns list of event dicts with magnitude, depth, time.
    """
    from datetime import datetime, timedelta
    end = datetime.utcnow()
    start = end - timedelta(days=days)

    url = (
        f"{settings.USGS_API_URL}query"
        f"?format=geojson"
        f"&latitude={latitude}&longitude={longitude}"
        f"&maxradiuskm={radius_km}"
        f"&starttime={start.strftime('%Y-%m-%d')}"
        f"&endtime={end.strftime('%Y-%m-%d')}"
        f"&minmagnitude=1.0"
        f"&orderby=time"
    )

    cache_key = f"seismicity:{latitude:.2f}:{longitude:.2f}:{radius_km}:{days}"
    cached = await cache_get(cache_key)
    if cached:
        return cached["events"]

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                if resp.status == 200:
                    data = await resp.json()
                else:
                    logger.error(f"USGS API returned status {resp.status}")
                    return []
    except Exception as exc:
        logger.error("USGS seismicity query failed: %s", exc)
        return []

    events = []
    for feat in data.get("features", []):
        props = feat["properties"]
        coords = feat["geometry"]["coordinates"]
        events.append({
            "event_id": feat["id"],
            "magnitude": props.get("mag"),
            "depth_km": coords[2] if len(coords) > 2 else None,
            "latitude": coords[1],
            "longitude": coords[0],
            "time": props.get("time"),
            "place": props.get("place"),
        })

    await cache_set(cache_key, {"events": events}, ttl=600)
    return events