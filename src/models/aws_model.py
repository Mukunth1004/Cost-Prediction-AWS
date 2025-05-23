# src/models/aws_model.py
from src.models.xgboost_model import XGBoostModel
from src.config import config
import numpy as np
import joblib
import os
from src.utils.logger import get_logger

logger = get_logger(__name__)

class AWSModel:
    def __init__(self):
        self.model = XGBoostModel()
        self.scaler = None
        
    def train(self, X_train, y_train, X_val, y_val):
        """Train XGBoost model"""
        return self.model.train(X_train, y_train, X_val, y_val)
        
    def predict(self, X):
        """Make predictions"""
        return self.model.predict(X)
        
    def save_model(self):
        """Scaler is saved separately during preprocessing"""
        pass
        
    def load_model(self):
        """Load trained model"""
        self.model.model = joblib.load(f"{config.MODEL_SAVE_PATH}/aws_xgb_model.pkl")
        self.scaler = joblib.load(f"{config.SCALER_SAVE_PATH}/aws_scaler.pkl")
        return True
    
    def _create_sequences(self, X, y):
        """Create time-series sequences"""
        X_seq, y_seq = [], []
        n_samples = len(X) - self.sequence_length
        if n_samples <= 0:
            raise ValueError(
                f"Not enough samples ({len(X)}) for sequence length {self.sequence_length}")
        for i in range(n_samples):
            X_seq.append(X[i:i+self.sequence_length])
            y_seq.append(y[i+self.sequence_length])
    
        return np.array(X_seq), np.array(y_seq)
    
    def _create_prediction_sequences(self, X):
        """Create sequences for prediction"""
        if len(X) < self.sequence_length:
            raise ValueError(f"Input data must have at least {self.sequence_length} time steps")
        
        # Use the last sequence_length data points
        return np.array([X[-self.sequence_length:]])