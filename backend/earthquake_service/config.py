# earthquake_service/config.py
"""Application configuration using pydantic-settings."""
from pydantic_settings import BaseSettings
from functools import lru_cache
from typing import Optional
from pathlib import Path
import os

# Get the absolute path to the backend directory
BACKEND_DIR = Path(__file__).parent.parent  # This is the backend directory
MODELS_DIR = BACKEND_DIR / "earthquake_service" / "models" / "saved"

class Settings(BaseSettings):
    APP_ENV: str = "development"
    SECRET_KEY: str = "change-me"

    # Database - SQLite
    DATABASE_URL: str = "sqlite+aiosqlite:///./earthquake.db"
    SYNC_DATABASE_URL: str = "sqlite:///./earthquake.db"
    
    # Redis
    REDIS_URL: str = "redis://localhost:6379/0"
    REDIS_TTL_WAVEFORM: int = 300
    REDIS_TTL_PREDICTION: int = 60
    USE_REDIS: bool = False

    # FDSN / IRIS
    IRIS_FDSN_URL: str = "https://service.iris.edu/fdsnws/event/1/"
    IRIS_STATION_URL: str = "https://service.iris.edu/fdsnws/station/1/"
    IRIS_DATASELECT_URL: str = "https://service.iris.edu/fdsnws/dataselect/1/"

    # USGS
    USGS_API_URL: str = "https://earthquake.usgs.gov/fdsnws/event/1/"

    # Model paths - Use absolute paths
    MODEL_DIR: str = str(MODELS_DIR)
    CNN_LSTM_MODEL: str = "cnn_lstm_model.pt"
    XGBOOST_MODEL: str = "xgb_classifier.json"
    SCALER_FILE: str = "feature_scaler.pkl"

    # Model metadata
    MODEL_VERSION: str = "1.0.0"
    MODEL_NAME: str = "cnn_lstm_xgb"

    # Prediction thresholds
    HIGH_RISK_THRESHOLD: float = 0.70
    MEDIUM_RISK_THRESHOLD: float = 0.40

    LOG_LEVEL: str = "INFO"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        # Disable protected namespace checking
        protected_namespaces = ()
        # Allow extra fields from env file
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()

# Print debug info
print(f"📁 Backend directory: {BACKEND_DIR}")
print(f"📁 Models directory: {settings.MODEL_DIR}")
print(f"📁 Models exist: {os.path.exists(settings.MODEL_DIR)}")