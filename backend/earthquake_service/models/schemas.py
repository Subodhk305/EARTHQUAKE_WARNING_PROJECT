"""Pydantic v2 request/response schemas."""
from __future__ import annotations
from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field


# ── Request schemas ────────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    latitude: float = Field(..., ge=-90, le=90, example=35.6762)
    longitude: float = Field(..., ge=-180, le=180, example=139.6503)
    location_name: Optional[str] = Field(None, example="Tokyo, Japan")
    radius_km: float = Field(200.0, ge=10, le=1000, description="Search radius for nearby historical events")
    include_waveform: bool = Field(True, description="Fetch IRIS waveform data for enhanced prediction")


# ── Response schemas ───────────────────────────────────────────────────────────

class PredictionResponse(BaseModel):
    request_id: str
    location: str
    latitude: float
    longitude: float
    probability: float = Field(..., ge=0, le=1)
    predicted_magnitude_class: str
    risk_level: str                 # Low | Medium | High
    confidence: float = Field(..., ge=0, le=1)
    magnitude_estimate: Optional[str] = None
    nearby_active_faults: int
    recent_seismicity_score: float
    model_version: str
    processing_time_ms: float
    timestamp: datetime


class EarthquakeEvent(BaseModel):
    event_id: str
    time: datetime
    latitude: float
    longitude: float
    depth_km: Optional[float]
    magnitude: float
    magnitude_type: Optional[str]
    place: Optional[str]
    magnitude_class: str


class HistoricalResponse(BaseModel):
    location: str
    latitude: float
    longitude: float
    radius_km: float
    total_events: int
    events: List[EarthquakeEvent]
    date_range_start: Optional[datetime]
    date_range_end: Optional[datetime]


class ModelMetrics(BaseModel):
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float
    training_samples: int
    evaluation_date: str


class ModelMetricsResponse(BaseModel):
    models: List[ModelMetrics]
    best_model: str
    confusion_matrix: List[List[int]]
    class_names: List[str]


class AlertMessage(BaseModel):
    alert_type: str         # HIGH_RISK | MODEL_UPDATE | SYSTEM
    location: str
    latitude: float
    longitude: float
    risk_level: str
    probability: float
    magnitude_class: str
    message: str
    timestamp: datetime
