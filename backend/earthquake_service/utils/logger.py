"""Structured logging configuration."""
import logging
import sys
from earthquake_service.config import settings


def setup_logging():
    log_level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    logging.basicConfig(
        level=log_level,
        format=fmt,
        handlers=[logging.StreamHandler(sys.stdout)],
    )
