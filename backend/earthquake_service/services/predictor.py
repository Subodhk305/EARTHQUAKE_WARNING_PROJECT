# earthquake_service/services/predictor.py
"""
Prediction Inference Service.

Full pipeline:
  1. Geocode location name to coordinates
  2. Fetch IRIS waveform
  3. Extract CNN-LSTM embedding
  4. Fetch recent seismicity (USGS)
  5. Build feature vector
  6. XGBoost prediction (using unified predict method)
  7. Map to risk level + confidence
  8. Log to DB
"""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Optional, Tuple
import aiohttp

import numpy as np
import torch

from earthquake_service.config import settings
from earthquake_service.models.schemas import PredictionResponse
from earthquake_service.services.model_loader import ModelLoader
from earthquake_service.services.iris_fetcher import fetch_waveform, get_recent_seismicity
from earthquake_service.services.feature_engineering import (
    build_feature_vector,
    MAG_CLASS_LABELS,
)
from earthquake_service.utils.cache import cache_get, cache_set

logger = logging.getLogger(__name__)

_DEVICE = ModelLoader.get_device()

# Geocoding cache
geocode_cache = {}


async def geocode_location(location_name: str) -> Optional[Tuple[float, float]]:
    """
    Convert location name to coordinates using OpenStreetMap Nominatim API.
    This is the proper way to get coordinates for any location.
    """
    if not location_name:
        return None
    
    # Check cache
    if location_name in geocode_cache:
        return geocode_cache[location_name]
    
    try:
        # Use OpenStreetMap Nominatim API (free, no API key needed)
        # Add "Maharashtra, India" context for better results
        search_query = f"{location_name}, Maharashtra, India"
        
        async with aiohttp.ClientSession() as session:
            url = "https://nominatim.openstreetmap.org/search"
            params = {
                "q": search_query,
                "format": "json",
                "limit": 1,
                "addressdetails": 0
            }
            headers = {
                "User-Agent": "EarthquakePredictionApp/1.0"
            }
            
            async with session.get(url, params=params, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    if data:
                        lat = float(data[0]["lat"])
                        lon = float(data[0]["lon"])
                        coords = (lat, lon)
                        geocode_cache[location_name] = coords
                        logger.info(f"✅ Geocoded '{location_name}' → ({lat}, {lon})")
                        return coords
                    else:
                        logger.warning(f"❌ No results for '{location_name}'")
                else:
                    logger.warning(f"Geocoding API returned {response.status}")
                    
    except Exception as e:
        logger.error(f"Geocoding error for '{location_name}': {e}")
    
    # If geocoding fails, try USGS geocoding as fallback
    try:
        async with aiohttp.ClientSession() as session:
            url = "https://earthquake.usgs.gov/fdsnws/event/1/query"
            params = {
                "format": "geojson",
                "limit": 1,
                "search": location_name
            }
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    if data.get("features"):
                        coords = data["features"][0]["geometry"]["coordinates"]
                        lat, lon = coords[1], coords[0]
                        geocode_cache[location_name] = (lat, lon)
                        logger.info(f"✅ USGS geocoded '{location_name}' → ({lat}, {lon})")
                        return (lat, lon)
    except Exception as e:
        logger.error(f"USGS geocoding error: {e}")
    
    return None


async def resolve_coordinates(
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
    location_name: Optional[str] = None
) -> Tuple[float, float, str]:
    """
    Resolve coordinates from either lat/lon or location name.
    Returns (latitude, longitude, display_name)
    """
    # If coordinates provided directly, use them
    if latitude is not None and longitude is not None:
        return latitude, longitude, f"{latitude:.4f}°, {longitude:.4f}°"
    
    # Otherwise, geocode the location name
    if location_name:
        coords = await geocode_location(location_name)
        if coords:
            lat, lon = coords
            logger.info(f"📍 Resolved '{location_name}' → ({lat}, {lon})")
            return lat, lon, location_name
    
    # If all else fails, use a reasonable default (center of India)
    logger.warning(f"⚠️ Could not resolve '{location_name}', using default coordinates")
    return 20.5937, 78.9629, "India (Center)"


def _risk_level(probability: float) -> str:
    if probability >= settings.HIGH_RISK_THRESHOLD:
        return "High"
    if probability >= settings.MEDIUM_RISK_THRESHOLD:
        return "Medium"
    return "Low"


def _waveform_to_embedding(waveform: np.ndarray) -> np.ndarray:
    """Run CNN-LSTM forward pass and return feature embedding."""
    model = ModelLoader.cnn_lstm
    if model is None:
        return np.zeros(128, dtype=np.float32)

    # Crop / pad to fixed length (60_000 samples = 600s × 100Hz)
    target = 60_000
    if waveform.shape[-1] > target:
        waveform = waveform[:, :target]
    elif waveform.shape[-1] < target:
        pad = target - waveform.shape[-1]
        waveform = np.pad(waveform, ((0, 0), (0, pad)))

    tensor = torch.tensor(waveform, dtype=torch.float32).unsqueeze(0).to(_DEVICE)
    with torch.no_grad():
        embedding = model.extract_features(tensor)
    return embedding.squeeze(0).cpu().numpy()


async def run_prediction(
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
    location_name: str = "",
    radius_km: float = 200.0,
    include_waveform: bool = True,
) -> PredictionResponse:
    """
    Run earthquake prediction for a given location.
    Can accept either coordinates or a location name.
    """
    t0 = time.perf_counter()
    request_id = str(uuid.uuid4())
    
    # Resolve coordinates
    lat, lon, display_location = await resolve_coordinates(
        latitude=latitude,
        longitude=longitude,
        location_name=location_name if location_name else None
    )
    
    # Use the display name
    if not location_name:
        location_name = display_location

    # ── Cache check ──────────────────────────────────────────────────────────
    cache_key = f"pred:{lat:.3f}:{lon:.3f}"
    cached = await cache_get(cache_key)
    if cached:
        logger.debug("Prediction cache HIT")
        cached["request_id"] = request_id
        cached["location"] = location_name
        cached["timestamp"] = datetime.now(tz=timezone.utc).isoformat()
        return PredictionResponse(**cached)

    # ── Fetch historical seismicity data first (this gives us real data) ────
    seismicity = await get_recent_seismicity(lat, lon, radius_km=radius_km, days=365)
    
    # If we have historical data, we can make a better prediction
    if seismicity:
        logger.info(f"📍 Found {len(seismicity)} historical events near {location_name}")
        avg_magnitude = np.mean([e.get("magnitude", 0) for e in seismicity if e.get("magnitude")])
        max_magnitude = max([e.get("magnitude", 0) for e in seismicity if e.get("magnitude")], default=0)
        recent_events = len([e for e in seismicity if e.get("magnitude", 0) > 3.0])
        
        logger.info(f"   Average magnitude: {avg_magnitude:.2f}, Max: {max_magnitude:.2f}")
        logger.info(f"   Recent significant events: {recent_events}")
    
    # ── Fetch waveform (optional) ────────────────────────────────────────────
    waveform = None
    if include_waveform:
        waveform = await fetch_waveform(lat, lon)
    
    if waveform is None:
        waveform = np.zeros((3, 60_000), dtype=np.float32)

    # ── Feature extraction ────────────────────────────────────────────────────
    embedding = _waveform_to_embedding(waveform)
    feat_vec = build_feature_vector(embedding, seismicity, lat, lon)

    # Scale if scaler is available
    if ModelLoader.scaler is not None:
        try:
            feat_vec = ModelLoader.scaler.transform(feat_vec.reshape(1, -1)).flatten()
        except Exception as e:
            logger.error(f"Scaling error: {e}")

    # ── XGBoost inference using unified predict method ───────────────────────
    probability = 0.05
    magnitude_class = "minor"
    confidence = 0.5
    risk = "Low"
    
    if ModelLoader.is_ready():
        try:
            # Use the unified predict method
            result = ModelLoader.predict(feat_vec.reshape(1, -1))
            
            # Extract results
            probability = result['probability']
            confidence = result['confidence']
            risk = result['risk_level']
            
            # Map risk level to magnitude class for response
            magnitude_map = {
                "Low Risk": "micro",
                "Medium Risk": "moderate",
                "High Risk": "strong"
            }
            magnitude_class = magnitude_map.get(result['class_name'], "minor")
            
            logger.info(f"✅ Prediction successful: {result['class_name']} with {confidence:.2%} confidence")
            
        except Exception as e:
            logger.error(f"XGBoost prediction error: {e}")
            # Fallback to heuristic
            if seismicity:
                recent_count = len([e for e in seismicity if e.get("magnitude", 0) >= 3.0])
                max_mag = max([e.get("magnitude", 0) for e in seismicity], default=0)
                probability = min(0.95, 0.05 + (recent_count / 100.0) + (max_mag / 50.0))
                
                if probability < 0.2:
                    magnitude_class = "micro"
                elif probability < 0.4:
                    magnitude_class = "minor"
                elif probability < 0.6:
                    magnitude_class = "moderate"
                elif probability < 0.8:
                    magnitude_class = "strong"
                else:
                    magnitude_class = "major"
                confidence = 0.6
                risk = _risk_level(probability)
    else:
        # Fallback heuristic when models not loaded
        logger.warning("Model not ready, using heuristic fallback")
        if seismicity:
            recent_count = len([e for e in seismicity if e.get("magnitude", 0) >= 3.0])
            max_mag = max([e.get("magnitude", 0) for e in seismicity], default=0)
            probability = min(0.95, 0.05 + (recent_count / 100.0) + (max_mag / 50.0))
            
            if probability < 0.2:
                magnitude_class = "micro"
            elif probability < 0.4:
                magnitude_class = "minor"
            elif probability < 0.6:
                magnitude_class = "moderate"
            elif probability < 0.8:
                magnitude_class = "strong"
            else:
                magnitude_class = "major"
            confidence = 0.6
            risk = _risk_level(probability)
        else:
            probability = 0.05
            magnitude_class = "micro"
            confidence = 0.3
            risk = "Low"

    # Magnitude estimate range string
    _mag_ranges = {
        "micro": "<2.0",
        "minor": "2.0–3.9",
        "moderate": "4.0–4.9",
        "strong": "5.0–5.9",
        "major": "6.0–6.9",
        "great": "≥7.0"
    }
    mag_estimate = _mag_ranges.get(magnitude_class, "unknown")

    # Count nearby active signals from seismicity
    nearby_faults = min(len([e for e in seismicity if (e.get("magnitude") or 0) >= 3.0]), 99)

    # Recent seismicity score 0-1
    recent_score = min(1.0, len(seismicity) / 100.0)

    elapsed_ms = (time.perf_counter() - t0) * 1000

    response = PredictionResponse(
        request_id=request_id,
        location=location_name,
        latitude=lat,
        longitude=lon,
        probability=round(probability, 4),
        predicted_magnitude_class=magnitude_class,
        risk_level=risk,
        confidence=round(confidence, 4),
        magnitude_estimate=mag_estimate,
        nearby_active_faults=nearby_faults,
        recent_seismicity_score=round(recent_score, 4),
        model_version=ModelLoader.model_version,
        processing_time_ms=round(elapsed_ms, 1),
        timestamp=datetime.now(tz=timezone.utc),
    )

    # Cache for 60 seconds
    await cache_set(cache_key, response.model_dump(mode="json"), ttl=settings.REDIS_TTL_PREDICTION)

    # Fire alert if high risk (non-blocking)
    if risk == "High":
        asyncio.create_task(_broadcast_alert(response))

    return response


async def _broadcast_alert(pred: PredictionResponse):
    """Broadcast high-risk alert via WebSocket manager."""
    try:
        from earthquake_service.websocket.manager import alert_manager
        from earthquake_service.models.schemas import AlertMessage
        alert = AlertMessage(
            alert_type="HIGH_RISK",
            location=pred.location,
            latitude=pred.latitude,
            longitude=pred.longitude,
            risk_level=pred.risk_level,
            probability=pred.probability,
            magnitude_class=pred.predicted_magnitude_class,
            message=(
                f"⚠️ HIGH RISK ALERT: {pred.location} — "
                f"{pred.probability*100:.1f}% probability of {pred.predicted_magnitude_class} event."
            ),
            timestamp=pred.timestamp,
        )
        await alert_manager.broadcast(alert.model_dump(mode="json"))
    except Exception as exc:
        logger.error("Alert broadcast failed: %s", exc)