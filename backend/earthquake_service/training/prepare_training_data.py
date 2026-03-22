"""
Prepare Training Data from ingested DB records + IRIS waveforms.

For each event in the DB:
  1. Fetch IRIS waveforms for the event time and nearest station.
  2. Build structured features from surrounding seismicity.
  3. Save to .npy files in the output directory.

Usage:
    python -m earthquake_service.training.prepare_training_data \
        --output-dir ./training_data \
        --max-events 10000
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import os

import numpy as np
from obspy.clients.fdsn import Client as FDSNClient
from obspy import UTCDateTime

sys.path.insert(0, ".")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger(__name__)

SAMPLE_RATE = 100
WINDOW_SECONDS = 600
N_CHANNELS = 3


def fetch_waveform_for_event(lat: float, lon: float, event_time: float) -> np.ndarray:
    """Fetch waveform around event time (synchronous, for batch processing)."""
    from earthquake_service.services.iris_fetcher import _stream_to_array, _normalize
    from obspy import UTCDateTime

    client = FDSNClient("IRIS")
    t_event = UTCDateTime(event_time)
    t_start = t_event - 60
    t_end = t_event + WINDOW_SECONDS

    target_samples = SAMPLE_RATE * WINDOW_SECONDS

    try:
        inventory = client.get_stations(
            latitude=lat, longitude=lon, maxradius=3.0,
            channel="HH?,BH?", level="channel",
            starttime=t_start, endtime=t_end,
        )
        stations = [(n.code, s.code) for n in inventory for s in n][:3]
        if not stations:
            return np.zeros((N_CHANNELS, target_samples), dtype=np.float32)

        st = client.get_waveforms(
            stations[0][0], stations[0][1], "*", "HH?,BH?", t_start, t_end
        )
        st.detrend("demean")
        st.detrend("linear")
        st.taper(0.05)
        st.filter("bandpass", freqmin=1.0, freqmax=40.0, corners=4, zerophase=True)
        st.resample(SAMPLE_RATE)
        st.merge(method=1, fill_value="interpolate")
        return _stream_to_array(st, target_samples)
    except Exception as exc:
        logger.debug("Waveform fetch failed: %s", exc)
        return np.zeros((N_CHANNELS, target_samples), dtype=np.float32)


async def prepare(output_dir: str, max_events: int):
    from earthquake_service.config import settings
    from earthquake_service.utils.database import Base, init_db
    from earthquake_service.models.db_models import EarthquakeRecord
    from earthquake_service.services.feature_engineering import (
        extract_geographic_features, magnitude_to_class, class_to_index
    )
    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
    from sqlalchemy import select
    import datetime

    os.makedirs(output_dir, exist_ok=True)

    engine = create_async_engine(settings.DATABASE_URL, echo=False)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)

    async with session_factory() as db:
        result = await db.execute(
            select(EarthquakeRecord)
            .order_by(EarthquakeRecord.time.desc())
            .limit(max_events)
        )
        records = result.scalars().all()

    logger.info("Processing %d events...", len(records))

    all_waveforms, all_structured, all_labels = [], [], []

    for i, rec in enumerate(records):
        geo = extract_geographic_features(rec.latitude, rec.longitude)

        # Placeholder structured features (seismicity computed from local window in production)
        structured = np.concatenate([
            np.zeros(25, dtype=np.float32),  # seismicity features — computed from surrounding events
            geo,
        ])

        label = class_to_index(magnitude_to_class(rec.magnitude))

        # Waveform (this is slow in batch — use parallel workers for production)
        if rec.time:
            ts = rec.time.timestamp()
            waveform = fetch_waveform_for_event(rec.latitude, rec.longitude, ts)
        else:
            waveform = np.zeros((N_CHANNELS, SAMPLE_RATE * WINDOW_SECONDS), dtype=np.float32)

        all_waveforms.append(waveform)
        all_structured.append(structured)
        all_labels.append(label)

        if (i + 1) % 100 == 0:
            logger.info("Processed %d / %d events", i + 1, len(records))

    np.save(os.path.join(output_dir, "waveforms.npy"), np.array(all_waveforms, dtype=np.float32))
    np.save(os.path.join(output_dir, "structured.npy"), np.array(all_structured, dtype=np.float32))
    np.save(os.path.join(output_dir, "labels.npy"), np.array(all_labels, dtype=np.int32))
    logger.info("✅ Saved training data to %s", output_dir)
    await engine.dispose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="./training_data")
    parser.add_argument("--max-events", type=int, default=10000)
    args = parser.parse_args()
    asyncio.run(prepare(args.output_dir, args.max_events))
