"""
Feature Engineering Module.

Combines:
  1. CNN-LSTM waveform embeddings (128-dim)
  2. Structured historical seismicity features (25-dim)
  3. Geographic features (10-dim)

Total feature vector: 163 dimensions → XGBoost input.
"""
from __future__ import annotations

import logging
import math
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import Optional

logger = logging.getLogger(__name__)

# Magnitude class boundaries (Richter scale)
MAG_CLASSES = {
    "micro": (0.0, 2.0),
    "minor": (2.0, 4.0),
    "moderate": (4.0, 5.0),
    "strong": (5.0, 6.0),
    "major": (6.0, 7.0),
    "great": (7.0, 10.0),
}
MAG_CLASS_LABELS = list(MAG_CLASSES.keys())

RISK_BINS = [0, 2.0, 4.0, 5.0, 6.0, 7.0, 10.0]


def magnitude_to_class(magnitude: float) -> str:
    for cls, (lo, hi) in MAG_CLASSES.items():
        if lo <= magnitude < hi:
            return cls
    return "great"


def class_to_index(cls: str) -> int:
    return MAG_CLASS_LABELS.index(cls) if cls in MAG_CLASS_LABELS else 1


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def extract_seismicity_features(events: list[dict]) -> np.ndarray:
    """
    Derive 25 statistical features from a list of recent seismic events.

    Features:
        [0]  total_count
        [1]  count_last_7d
        [2]  count_last_30d
        [3-7] magnitude percentiles (10,25,50,75,90)
        [8]  max_magnitude
        [9]  mean_magnitude
        [10] std_magnitude
        [11] energy_sum_log (sum of 10^(1.5*M))
        [12] b_value (Gutenberg–Richter)
        [13] mean_depth_km
        [14] std_depth_km
        [15] temporal_rate_7d      (events/day)
        [16] temporal_rate_30d
        [17] acceleration_index    (rate_7d / rate_30d)
        [18] count_m2plus
        [19] count_m4plus
        [20] count_m6plus
        [21] time_since_last_s     (seconds since most recent event)
        [22] cluster_dispersion_km (std of distances from centroid)
        [23] shallow_fraction      (depth < 35 km)
        [24] aftershock_decay      (basic Omori-type recency weighting)
    """
    n = 25
    if not events:
        return np.zeros(n, dtype=np.float32)

    now = datetime.now(tz=timezone.utc)
    mags, depths, times, lats, lons = [], [], [], [], []

    for ev in events:
        m = ev.get("magnitude")
        d = ev.get("depth_km")
        t_ms = ev.get("time")
        lat = ev.get("latitude")
        lon = ev.get("longitude")
        if m is not None:
            mags.append(float(m))
        if d is not None:
            depths.append(float(d))
        if t_ms is not None:
            try:
                ts = datetime.fromtimestamp(t_ms / 1000, tz=timezone.utc)
                times.append(ts)
            except Exception:
                pass
        if lat is not None and lon is not None:
            lats.append(float(lat))
            lons.append(float(lon))

    mags = np.array(mags, dtype=np.float32)
    depths = np.array(depths, dtype=np.float32)
    now7 = now - timedelta(days=7)
    now30 = now - timedelta(days=30)

    feats = np.zeros(n, dtype=np.float32)
    feats[0] = len(events)
    feats[1] = sum(1 for t in times if t >= now7)
    feats[2] = sum(1 for t in times if t >= now30)

    if len(mags):
        for i, pct in enumerate([10, 25, 50, 75, 90]):
            feats[3 + i] = float(np.percentile(mags, pct))
        feats[8] = float(mags.max())
        feats[9] = float(mags.mean())
        feats[10] = float(mags.std()) if len(mags) > 1 else 0.0
        # Energy (proxy): log10(sum(10^(1.5*M)))
        energy = np.sum(10 ** (1.5 * mags))
        feats[11] = float(np.log10(energy + 1e-10))
        # b-value (MLE): b = log10(e) / (mean_M - Mc), Mc = min observed
        if len(mags) > 1:
            Mc = float(mags.min())
            mean_excess = float(mags.mean()) - Mc
            if mean_excess > 0:
                feats[12] = float(np.log10(math.e) / mean_excess)
        feats[18] = float(np.sum(mags >= 2.0))
        feats[19] = float(np.sum(mags >= 4.0))
        feats[20] = float(np.sum(mags >= 6.0))

    if len(depths):
        feats[13] = float(depths.mean())
        feats[14] = float(depths.std()) if len(depths) > 1 else 0.0
        feats[23] = float(np.sum(depths < 35) / len(depths))

    days7 = max(feats[1], 0)
    days30 = max(feats[2], 0)
    feats[15] = days7 / 7.0
    feats[16] = days30 / 30.0
    feats[17] = feats[15] / (feats[16] + 1e-8)

    if times:
        most_recent = max(times)
        feats[21] = float((now - most_recent).total_seconds())
        # Omori-type aftershock decay: sum of 1/(1+t_hours)
        decay = sum(1.0 / (1.0 + (now - t).total_seconds() / 3600.0) for t in times)
        feats[24] = float(decay)

    if len(lats) > 1:
        c_lat = np.mean(lats)
        c_lon = np.mean(lons)
        dists = [haversine_km(c_lat, c_lon, la, lo) for la, lo in zip(lats, lons)]
        feats[22] = float(np.std(dists))

    return feats


def extract_geographic_features(latitude: float, longitude: float) -> np.ndarray:
    """
    10 geographic / tectonic proxy features.
    In production these can be enriched with actual plate boundary distances,
    fault databases, or isostatic anomaly data.

    Features:
        [0]  latitude  (normalised -1 to 1)
        [1]  longitude (normalised -1 to 1)
        [2]  sin(latitude_rad)
        [3]  cos(latitude_rad)
        [4]  sin(longitude_rad)
        [5]  cos(longitude_rad)
        [6]  distance_to_ring_of_fire_proxy (degrees from Pacific rim centroid)
        [7]  is_coastal (rough heuristic)
        [8]  latitude_abs
        [9]  lon_band (10-degree bucket normalised)
    """
    feats = np.zeros(10, dtype=np.float32)
    feats[0] = latitude / 90.0
    feats[1] = longitude / 180.0
    lat_r = math.radians(latitude)
    lon_r = math.radians(longitude)
    feats[2] = math.sin(lat_r)
    feats[3] = math.cos(lat_r)
    feats[4] = math.sin(lon_r)
    feats[5] = math.cos(lon_r)
    # Pacific centroid proxy: (0°N, -160°W) for Ring of Fire
    feats[6] = haversine_km(latitude, longitude, 0.0, -160.0) / 10000.0
    feats[7] = 1.0 if abs(latitude) < 60 and abs(longitude) > 100 else 0.0  # rough
    feats[8] = abs(latitude) / 90.0
    feats[9] = ((longitude + 180) // 10) / 36.0
    return feats


def build_feature_vector(
    waveform_embedding: np.ndarray,
    seismicity_events: list[dict],
    latitude: float,
    longitude: float,
) -> np.ndarray:
    """
    Concatenate all features into a single 1-D numpy array for XGBoost.

    Args:
        waveform_embedding: (128,) float32 from CNN-LSTM
        seismicity_events:  list of recent event dicts
        latitude, longitude: target location

    Returns:
        np.ndarray of shape (163,)
    """
    seism_feats = extract_seismicity_features(seismicity_events)    # 25
    geo_feats = extract_geographic_features(latitude, longitude)    # 10
    feat_vec = np.concatenate([waveform_embedding, seism_feats, geo_feats])  # 163
    return feat_vec.astype(np.float32)
