"""
USGS Earthquake Catalog Downloader.

Downloads historical earthquake data from USGS FDSN API and stores it in
PostgreSQL. Fetches in 6-month chunks to avoid API limits.

Usage:
    python -m earthquake_service.training.ingest_data \
        --start-year 2000 --end-year 2023 --min-magnitude 2.0
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from datetime import datetime, date, timedelta

import aiohttp
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.dialects.postgresql import insert as pg_insert

sys.path.insert(0, ".")

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger(__name__)


async def fetch_usgs_chunk(
    session: aiohttp.ClientSession,
    start: date,
    end: date,
    min_mag: float,
    usgs_url: str,
) -> list[dict]:
    url = (
        f"{usgs_url}query?format=geojson"
        f"&starttime={start}&endtime={end}"
        f"&minmagnitude={min_mag}"
        f"&orderby=time-asc"
        f"&limit=20000"
    )
    async with session.get(url, timeout=aiohttp.ClientTimeout(total=60)) as resp:
        resp.raise_for_status()
        data = await resp.json()
    return data.get("features", [])


def feature_to_row(feat: dict) -> dict:
    from earthquake_service.services.feature_engineering import magnitude_to_class
    props = feat["properties"]
    coords = feat["geometry"]["coordinates"]
    mag = props.get("mag") or 0.0
    t_ms = props.get("time")
    ts = datetime.utcfromtimestamp(t_ms / 1000) if t_ms else None
    return {
        "event_id": feat["id"],
        "time": ts,
        "latitude": coords[1],
        "longitude": coords[0],
        "depth_km": coords[2] if len(coords) > 2 else None,
        "magnitude": mag,
        "magnitude_type": props.get("magType"),
        "place": props.get("place"),
        "magnitude_class": magnitude_to_class(mag),
        "source": "USGS",
        "raw_properties": props,
    }


async def ingest(start_year: int, end_year: int, min_magnitude: float):
    from earthquake_service.config import settings
    from earthquake_service.utils.database import Base
    from earthquake_service.models.db_models import EarthquakeRecord  # noqa

    engine = create_async_engine(settings.DATABASE_URL, echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(engine, expire_on_commit=False)

    chunk_start = date(start_year, 1, 1)
    chunk_end = date(end_year, 12, 31)
    delta = timedelta(days=180)

    total_inserted = 0
    async with aiohttp.ClientSession() as http:
        current = chunk_start
        while current < chunk_end:
            nxt = min(current + delta, chunk_end)
            logger.info("Fetching %s → %s ...", current, nxt)
            try:
                features = await fetch_usgs_chunk(http, current, nxt, min_magnitude, settings.USGS_API_URL)
            except Exception as exc:
                logger.error("Failed chunk %s–%s: %s", current, nxt, exc)
                current = nxt + timedelta(days=1)
                continue

            rows = [feature_to_row(f) for f in features]
            if rows:
                async with session_factory() as db:
                    stmt = (
                        pg_insert(EarthquakeRecord.__table__)
                        .values(rows)
                        .on_conflict_do_nothing(index_elements=["event_id"])
                    )
                    result = await db.execute(stmt)
                    await db.commit()
                    total_inserted += result.rowcount or 0
                    logger.info("Inserted %d rows (chunk total: %d)", result.rowcount or 0, total_inserted)

            current = nxt + timedelta(days=1)

    logger.info("✅ Ingestion complete. Total rows inserted: %d", total_inserted)
    await engine.dispose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-year", type=int, default=2000)
    parser.add_argument("--end-year", type=int, default=2023)
    parser.add_argument("--min-magnitude", type=float, default=2.0)
    args = parser.parse_args()
    asyncio.run(ingest(args.start_year, args.end_year, args.min_magnitude))
