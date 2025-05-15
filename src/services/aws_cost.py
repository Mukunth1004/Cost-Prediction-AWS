# src/services/aws_cost.py
from datetime import date, timedelta
from typing import Dict, List
import numpy as np
import pandas as pd
import joblib
from src.ml.data_preprocessing import DataPreprocessor
from src.ml.feature_engineering import FeatureEngineer
from src.config.settings import settings

class AWSCostService:
    def __init__(self):
        self.model = self._load_model()
        self.preprocessor = DataPreprocessor()
        self.feature_engineer = FeatureEngineer()
        
    def _load_model(self):
        try:
            return joblib.load(settings.MODEL_PATH)
        except FileNotFoundError:
            raise Exception("Model not found. Please train the model first.")
            
    def get_available_services(self) -> List[str]:
        return [
            "AmazonEC2", "AmazonS3", "AmazonRDS", "AmazonDynamoDB",
            "AmazonLambda", "AmazonCloudFront", "AmazonAPIGateway",
            "AWSIoT", "AmazonKinesis", "AmazonSNS"
        ]
        
    def predict_cost(self, services: List[str], start_date: date, end_date: date, 
                   historical_data: Dict[str, List[float]]) -> Dict:
        """
        Predict costs for multiple services over a date range
        
        Args:
            services: List of AWS services to predict
            start_date: Start date of prediction period
            end_date: End date of prediction period
            historical_data: Dictionary mapping services to their historical costs
            
        Returns:
            Dictionary containing predictions and cost breakdown
        """
        # Create proper 3D input
        sequences = []
        for service in services:
            history = historical_data.get(service, [0.0]*30)[-30:]  # Last 30 days
            service_seq = self._create_features_for_service(history)
            sequences.append(service_seq)
    
        X_seq = np.array(sequences)  # Shape: (n_services, 30, 16)
        X_tab = self._create_tabular_features(start_date, end_date, len(services))
        
        # Predict and format results
        predictions = self.model.predict(X_seq, X_tab)
        return self._format_predictions(services, predictions, start_date, end_date)
    
    def _create_features_for_service(self, history: List[float]) -> np.ndarray:
        """
        Create all 16 time-series features from historical cost data
        
        Args:
            history: List of 30 historical cost values
            
        Returns:
            numpy array of shape (30, 16) containing all features
        """
        if len(history) != 30:
            raise ValueError(f"History must contain exactly 30 values, got {len(history)}")
            
        # Convert to pandas Series for rolling operations
        s = pd.Series(history)
        
        # Initialize feature matrix (30 timesteps × 16 features)
        features = np.zeros((30, 16))
        
        # 1. Original values (feature 0)
        features[:, 0] = history
        
        # 2. Lag features (features 1-5)
        for i, lag in enumerate([1, 3, 7, 14, 30], start=1):
            features[:, i] = s.shift(lag).fillna(0).values
            
        # 3. Rolling statistics (features 6-13)
        for i, window in enumerate([3, 7, 14, 30], start=6):
            # Mean
            features[:, i] = s.rolling(window=window).mean().fillna(0).values
            # Std
            features[:, i+4] = s.rolling(window=window).std().fillna(0).values
            
        # 4. Exponential moving averages (features 14-16)
        for i, alpha in enumerate([0.1, 0.3, 0.5], start=14):
            features[:, i] = s.ewm(alpha=alpha).mean().values
            
        return features
    
    def _create_tabular_features(self, start_date: date, end_date: date, 
                               n_services: int) -> np.ndarray:
        """
        Create tabular features for the date range
        
        Args:
            start_date: Start date of prediction period
            end_date: End date of prediction period
            n_services: Number of services being predicted
            
        Returns:
            numpy array of tabular features
        """
        date_range = pd.date_range(start_date, end_date)
        tab_features = []
        
        for day in date_range:
            # Basic date features
            features = [
                day.day,
                day.month,
                day.year,
                day.dayofweek,
                day.dayofyear,
                day.weekofyear
            ]
            tab_features.append(features)
            
        # Repeat for each service and stack
        return np.tile(np.array(tab_features), (n_services, 1))
    
    def _format_predictions(self, services: List[str], predictions: np.ndarray,
                          start_date: date, end_date: date) -> Dict:
        """
        Format raw predictions into API response format
        
        Args:
            services: List of service names
            predictions: Raw prediction array
            start_date: Start date of prediction period
            end_date: End date of prediction period
            
        Returns:
            Formatted prediction dictionary
        """
        date_range = pd.date_range(start_date, end_date)
        result = {
            "predictions": {},
            "total_cost": 0.0,
            "cost_breakdown": {}
        }
        
        for i, service in enumerate(services):
            service_preds = predictions[i].tolist()
            result["predictions"][service] = service_preds
            result["total_cost"] += sum(service_preds)
            
        # Calculate cost breakdown percentages
        for service in services:
            service_cost = sum(result["predictions"][service])
            result["cost_breakdown"][service] = (
                service_cost / result["total_cost"] * 100
            )
            
        return result