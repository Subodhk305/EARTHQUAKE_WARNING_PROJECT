import os
import re

def fix_routes():
    """Add settings import to routes.py"""
    routes_path = "earthquake_service/api/routes.py"
    with open(routes_path, 'r') as f:
        content = f.read()
    
    if "from earthquake_service.config import settings" not in content:
        # Add import after other imports
        content = re.sub(
            r'(from fastapi import.*?\n)',
            r'\1from earthquake_service.config import settings\n',
            content
        )
        with open(routes_path, 'w') as f:
            f.write(content)
        print("✅ Added settings import to routes.py")

def fix_historical_endpoint():
    """Add query parameter endpoint for historical data"""
    routes_path = "earthquake_service/api/routes.py"
    with open(routes_path, 'r') as f:
        content = f.read()
    
    endpoint_code = '''
@router.get("/historical", response_model=HistoricalResponse, tags=["Historical"])
async def get_historical_query(
    location: str = Query(..., description="Location name"),
    lat: float = Query(..., ge=-90, le=90),
    lon: float = Query(..., ge=-180, le=180),
    radius_km: float = Query(200.0, ge=10, le=2000),
    days: int = Query(365, ge=1, le=3650),
    db: AsyncSession = Depends(get_db),
):
    """Return historical events near a location (query parameter version)."""
    # Approximate bounding box (1° ≈ 111 km)
    delta = radius_km / 111.0
    since = datetime.now(tz=timezone.utc) - timedelta(days=days)

    stmt = (
        select(EarthquakeRecord)
        .where(
            EarthquakeRecord.latitude.between(lat - delta, lat + delta),
            EarthquakeRecord.longitude.between(lon - delta, lon + delta),
            EarthquakeRecord.time >= since.replace(tzinfo=None),
        )
        .order_by(EarthquakeRecord.time.desc())
        .limit(500)
    )
    rows = (await db.execute(stmt)).scalars().all()

    events = [
        EarthquakeEvent(
            event_id=r.event_id,
            time=r.time,
            latitude=r.latitude,
            longitude=r.longitude,
            depth_km=r.depth_km,
            magnitude=r.magnitude,
            magnitude_type=r.magnitude_type,
            place=r.place,
            magnitude_class=r.magnitude_class or magnitude_to_class(r.magnitude),
        )
        for r in rows
    ]

    response = HistoricalResponse(
        location=location,
        latitude=lat,
        longitude=lon,
        radius_km=radius_km,
        total_events=len(events),
        events=events,
        date_range_start=events[-1].time if events else None,
        date_range_end=events[0].time if events else None,
    )
    return response
'''
    
    if "@router.get(\"/historical\"" not in content:
        content += endpoint_code
        with open(routes_path, 'w') as f:
            f.write(content)
        print("✅ Added query parameter historical endpoint")

if __name__ == "__main__":
    fix_routes()
    fix_historical_endpoint()
    print("🎉 Fixes applied! Restart the server.")