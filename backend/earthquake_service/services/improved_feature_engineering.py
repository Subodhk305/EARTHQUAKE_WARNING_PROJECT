# earthquake_service/services/improved_feature_engineering.py
"""
Enhanced feature engineering with better temporal and spatial features.
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Any

def extract_enhanced_features(
    seismicity_data: List[Dict],
    latitude: float,
    longitude: float,
    waveform_features: np.ndarray = None
) -> np.ndarray:
    """
    Extract enhanced features for earthquake prediction
    """
    features = []
    
    # Basic location features
    features.extend([latitude, longitude])
    
    if seismicity_data:
        # Create DataFrame for easier analysis
        df = pd.DataFrame(seismicity_data)
        
        # Temporal features
        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'])
            df['days_since'] = (datetime.now() - df['time']).dt.days
            
            # Recency weighting (more recent events weighted higher)
            weights = np.exp(-df['days_since'] / 30)  # 30-day decay
            weighted_magnitudes = df['magnitude'].values * weights
            
            # Weighted statistics
            features.append(np.sum(weighted_magnitudes))  # Weighted sum
            features.append(np.mean(weighted_magnitudes))  # Weighted mean
            features.append(np.max(weighted_magnitudes))  # Max weighted
        
        # Magnitude statistics
        magnitudes = df['magnitude'].values
        features.extend([
            np.mean(magnitudes),  # Mean magnitude
            np.std(magnitudes),   # Std of magnitude
            np.max(magnitudes),   # Max magnitude
            np.percentile(magnitudes, 90),  # 90th percentile
            np.sum(magnitudes > 4.0),  # Count of M4+ events
            np.sum(magnitudes > 5.0),  # Count of M5+ events
            np.sum(magnitudes > 6.0),  # Count of M6+ events
        ])
        
        # Temporal clustering
        if len(magnitudes) > 1:
            time_diffs = np.diff(df['time'].values.astype('int64')) / 1e9 / (24*3600)  # in days
            features.extend([
                np.mean(time_diffs),  # Mean time between events
                np.std(time_diffs),   # Std of time differences
                np.min(time_diffs),   # Minimum time between events
            ])
        else:
            features.extend([0, 0, 0])
        
        # Spatial clustering
        if 'latitude' in df.columns and 'longitude' in df.columns:
            distances = np.sqrt(
                (df['latitude'] - latitude)**2 + 
                (df['longitude'] - longitude)**2
            )
            features.extend([
                np.mean(distances),  # Mean distance
                np.std(distances),   # Std of distances
                np.min(distances),   # Minimum distance
                np.sum(distances < 1.0),  # Events within 1 degree
            ])
        else:
            features.extend([0, 0, 0, 0])
        
        # Rate of seismicity
        for window in [7, 30, 90, 365]:  # 1 week, 1 month, 3 months, 1 year
            recent = df[df['days_since'] <= window]
            features.append(len(recent))  # Event count in window
        
    else:
        # No historical data - add zeros
        features.extend([0] * 20)  # Adjust number based on features added
    
    # Add waveform features if available
    if waveform_features is not None:
        features.extend(waveform_features.tolist())
    
    return np.array(features, dtype=np.float32)