# src/utils/helpers.py
import numpy as np
import pandas as pd
from typing import Dict, List

def convert_to_dataframe(data: List[Dict]) -> pd.DataFrame:
    """Convert list of dictionaries to pandas DataFrame"""
    return pd.DataFrame(data)

def calculate_cost_components(features: Dict, coefficients: Dict) -> Dict:
    """Calculate cost components based on features and coefficients"""
    components = {}
    for feature, value in features.items():
        if feature in coefficients:
            components[feature] = value * coefficients[feature]
    return components

def generate_time_features(timestamps: List[str]) -> Dict:
    """Generate time-based features from timestamps"""
    df = pd.DataFrame({'timestamp': pd.to_datetime(timestamps)})
    return {
        'hour': df['timestamp'].dt.hour.values,
        'day_of_week': df['timestamp'].dt.dayofweek.values,
        'month': df['timestamp'].dt.month.values
    }