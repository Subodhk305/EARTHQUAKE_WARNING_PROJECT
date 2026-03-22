"""SQLAlchemy ORM models for earthquake records and prediction logs."""
import datetime
from sqlalchemy import Column, Integer, Float, String, DateTime, JSON, Text
from earthquake_service.utils.database import Base


class EarthquakeRecord(Base):
    """Historical earthquake event from USGS catalog."""
    __tablename__ = "earthquake_records"

    id = Column(Integer, primary_key=True, index=True)
    event_id = Column(String(64), unique=True, index=True, nullable=False)
    time = Column(DateTime, nullable=False, index=True)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    depth_km = Column(Float)
    magnitude = Column(Float, nullable=False)
    magnitude_type = Column(String(16))
    place = Column(Text)
    magnitude_class = Column(String(16))   # minor/moderate/strong/major/great
    source = Column(String(32), default="USGS")
    raw_properties = Column(JSON)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)


class PredictionLog(Base):
    """Stores every prediction request + result for audit and retraining."""
    __tablename__ = "prediction_logs"

    id = Column(Integer, primary_key=True, index=True)
    request_id = Column(String(64), unique=True, index=True)
    latitude = Column(Float)
    longitude = Column(Float)
    location_name = Column(String(256))
    probability = Column(Float)
    predicted_magnitude_class = Column(String(32))
    risk_level = Column(String(16))
    confidence = Column(Float)
    model_version = Column(String(32))
    features_used = Column(JSON)
    prediction_time_ms = Column(Float)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
