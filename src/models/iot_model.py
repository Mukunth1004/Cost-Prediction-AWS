# src/models/iot_model.py
import joblib
import os
import numpy as np
from src.models.xgboost_model import XGBoostModel
from src.config import config
from src.utils.logger import get_logger

logger = get_logger(__name__)

class IoTModel:
    def __init__(self):
        self.model = XGBoostModel()
        self.scaler = None
        self.is_trained = False
    
    def train(self, X_train, y_train, X_val, y_val):
        """Train the IoT cost prediction model using XGBoost"""
        try:
            logger.info("Training IoT cost prediction model")
            
            # Train XGBoost model
            result = self.model.train(X_train, y_train, X_val, y_val)
            
            if result['status'] == 'success':
                self.is_trained = True
                self.save_model()
            
            return result
            
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            return {'status': 'error', 'message': str(e)}
    
    def predict(self, X):
        """Ensure proper input shape for XGBoost"""
        if not self.is_trained:
            self.load_model()
    
    # Reshape if coming from sequence data
        if len(X.shape) == 3:  # If (samples, timesteps, features)
            X = X.reshape(X.shape[0], -1)  # Flatten time dimension
    
        return self.model.predict(X)
    
    def save_model(self):
        """Save the trained model and scaler"""
        if not os.path.exists(config.MODEL_SAVE_PATH):
            os.makedirs(config.MODEL_SAVE_PATH)
        
        # Save XGBoost model
        joblib.dump(self.model.model, f"{config.MODEL_SAVE_PATH}/iot_xgb_model.pkl")
        
        # Save scaler if it exists
        if self.scaler is not None:
            if not os.path.exists(config.SCALER_SAVE_PATH):
                os.makedirs(config.SCALER_SAVE_PATH)
            joblib.dump(self.scaler, f"{config.SCALER_SAVE_PATH}/iot_scaler.pkl")
    
    def load_model(self):
        """Load a trained model"""
        try:
            self.model.model = joblib.load(f"{config.MODEL_SAVE_PATH}/iot_xgb_model.pkl")
            self.scaler = joblib.load(f"{config.SCALER_SAVE_PATH}/iot_scaler.pkl")
            self.is_trained = True
            return True
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            self.is_trained = False
            return False
    
    def evaluate(self, X, y):
        """Evaluate model performance"""
        if not self.is_trained:
            raise ValueError("Model not trained yet. Call train() first.")
        return self.model.evaluate(X, y)
    
    def _create_sequences(self, X, y):
        """Create time-series sequences"""
        X_seq = []
        y_seq = []
        for i in range(len(X) - self.sequence_length):
            X_seq.append(X[i:i+self.sequence_length])
            y_seq.append(y[i+self.sequence_length])
        return np.array(X_seq), np.array(y_seq)
    
    def _create_prediction_sequences(self, X):
        """Create sequences for prediction"""
        if len(X) < self.sequence_length:
            raise ValueError(f"Input data must have at least {self.sequence_length} time steps")
        
        # Use the last sequence_length data points
        return np.array([X[-self.sequence_length:]])