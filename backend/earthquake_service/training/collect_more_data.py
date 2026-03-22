# earthquake_service/training/collect_more_data.py
"""
Collect more comprehensive training data from USGS.
"""
import aiohttp
import asyncio
import pandas as pd
from datetime import datetime, timedelta

async def collect_comprehensive_data():
    """Collect more comprehensive earthquake data"""
    
    base_url = "https://earthquake.usgs.gov/fdsnws/event/1/query"
    
    # Collect data for last 20 years
    start_time = datetime.now() - timedelta(days=365*20)
    end_time = datetime.now()
    
    params = {
        "format": "geojson",
        "starttime": start_time.strftime("%Y-%m-%d"),
        "endtime": end_time.strftime("%Y-%m-%d"),
        "minmagnitude": 2.5,  # Include smaller earthquakes
        "maxmagnitude": 9.0,
        "limit": 20000  # Get up to 20k events
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.get(base_url, params=params) as response:
            data = await response.json()
            
            events = []
            for feature in data['features']:
                props = feature['properties']
                geometry = feature['geometry']
                
                event = {
                    'time': props['time'],
                    'latitude': geometry['coordinates'][1],
                    'longitude': geometry['coordinates'][0],
                    'depth': geometry['coordinates'][2],
                    'magnitude': props['mag'],
                    'mag_type': props['magType'],
                    'place': props['place'],
                    'event_id': props['id']
                }
                events.append(event)
            
            df = pd.DataFrame(events)
            df.to_csv('earthquake_training_data.csv', index=False)
            print(f"✅ Collected {len(df)} events")
            return df

if __name__ == "__main__":
    asyncio.run(collect_comprehensive_data())