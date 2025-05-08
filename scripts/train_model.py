import sys
import os
import numpy as np  # <-- Added import
import pandas as pd
from pathlib import Path

# Absolute path resolution
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
core_path = project_root / "core"

# Add both project root and core directory to path
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(core_path))

from core.models.hybrid_model import HybridCostPredictor
from core.config import settings

def load_sample_data():
    """Load or generate sample data if no real data exists"""
    dates = pd.date_range(start='2020-01-01', end='2023-01-01', freq='D')
    data = {
        'date': dates,
        'ec2_usage': [max(0, min(100, 50 + i * 0.1 + (10 if i % 7 == 0 else 0))) for i in range(len(dates))],
        's3_usage': [max(0, min(200, 80 + i * 0.05 + (20 if i % 30 == 0 else 0))) for i in range(len(dates))],
        'lambda_usage': [max(0, min(300, 100 + i * 0.2 + (50 if i % 14 == 0 else 0))) for i in range(len(dates))],
        'rds_usage': [max(0, min(150, 60 + i * 0.08 + (15 if i % 21 == 0 else 0))) for i in range(len(dates))],
        'cost': [0] * len(dates)
    }
    
    # Calculate synthetic cost
    df = pd.DataFrame(data)
    spike_multiplier = np.where(df.index % 30 == 0, 1.2, 1.0)
    df['cost'] = (
        df['ec2_usage'] * 0.05 +
        df['s3_usage'] * 0.023 +
        df['lambda_usage'] * 0.0000166667 +
        df['rds_usage'] * 0.025
    ) * spike_multiplier
    
    return df

def main():
    # Load data
    if os.path.exists(settings.data_path):
        data = pd.read_csv(settings.data_path)
    else:
        print("No data found at configured path. Using sample data.")
        data = load_sample_data()
        os.makedirs(os.path.dirname(settings.data_path), exist_ok=True)
        data.to_csv(settings.data_path, index=False)
    
    # Train model
    model = HybridCostPredictor()
    model.train(data)
    # In your train_model.py, add this before saving:
    os.makedirs(os.path.dirname(settings.model_path), exist_ok=True)
    # Save model
    model.save(settings.model_path)
    print(f"Model trained and saved to {settings.model_path}")

if __name__ == "__main__":
    main()