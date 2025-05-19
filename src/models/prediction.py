import numpy as np
import pandas as pd
import tensorflow as tf
from xgboost import XGBRegressor
from src.models.data_processing import DataProcessor
from src.config.settings import settings

class AWSCostPredictor:
    def __init__(self):
        self.processor = DataProcessor()
        self.lstm_model = tf.keras.models.load_model(settings.AWS_MODEL_PATH)
        self.xgb_model = XGBRegressor()
        self.xgb_model.load_model(settings.XGBOOST_MODEL_PATH)
        self.n_steps = 10
        
    def predict(self, input_data):
        """Predict AWS costs with new features"""
        if isinstance(input_data, pd.DataFrame):
            processed_data, _, _ = self.processor.load_aws_data(input_data)
        else:
            processed_data = self.processor.scaler.transform(input_data)
            
        if len(processed_data) < self.n_steps:
            raise ValueError(f"Need at least {self.n_steps} time steps")
            
        X = processed_data[-self.n_steps:]
        X = np.expand_dims(X, axis=0)
        
        # LSTM prediction
        lstm_pred = self.lstm_model.predict(X).flatten()[0]
        
        # XGBoost prediction
        X_flat = X.reshape(1, -1)
        xgb_pred = self.xgb_model.predict(X_flat)[0]
        
        return (lstm_pred + xgb_pred) / 2
    
    def explain_prediction(self, input_data):
        """Explain cost breakdown"""
        if isinstance(input_data, pd.DataFrame):
            features = input_data.iloc[-1]
        else:
            features = input_data
            
        breakdown = [
            {
                "feature": "EC2 Instance Hours",
                "value": features['ec2_instance_hours'],
                "impact": features['ec2_instance_hours'] * 0.05  # Example impact factor
            },
            # Add other features similarly
        ]
        
        return breakdown

class IoTCostPredictor:
    def __init__(self):
        self.processor = DataProcessor()
        self.model = tf.keras.models.load_model(settings.IOT_MODEL_PATH)
        self.n_steps = 10
        
    def predict(self, input_data):
        """Predict IoT costs with new features"""
        if isinstance(input_data, pd.DataFrame):
            processed_data, _, _ = self.processor.load_iot_data(input_data)
        else:
            processed_data = self.processor.scaler.transform(input_data)
            
        if len(processed_data) < self.n_steps:
            raise ValueError(f"Need at least {self.n_steps} time steps")
            
        X = processed_data[-self.n_steps:]
        X = np.expand_dims(X, axis=0)
        
        return self.model.predict(X).flatten()[0]
    
    def explain_prediction(self, input_data):
        """Explain IoT cost breakdown with rules and policies"""
        if isinstance(input_data, pd.DataFrame):
            features = input_data.iloc[-1]
        else:
            features = input_data
            
        # Calculate cost contributions
        rule_cost = features['num_rules'] * 0.10  # Example $0.10 per rule
        policy_cost = features['num_policies'] * 0.05  # Example $0.05 per policy
        message_cost = features['mqtt_messages_per_day'] * 0.0001  # Example $0.0001 per message
        
        breakdown = [
            {
                "component": "Rules",
                "count": features['num_rules'],
                "cost": rule_cost,
                "explanation": f"{features['num_rules']} rules attached at $0.10 per rule"
            },
            {
                "component": "Policies",
                "count": features['num_policies'],
                "cost": policy_cost,
                "explanation": f"{features['num_policies']} policies attached at $0.05 per policy"
            },
            {
                "component": "MQTT Messages",
                "count": features['mqtt_messages_per_day'],
                "cost": message_cost,
                "explanation": f"{features['mqtt_messages_per_day']} messages at $0.0001 per message"
            }
        ]
        
        return breakdown
    
    def get_optimizations(self, input_data):
        """Get IoT-specific optimizations"""
        optimizations = []
        
        if input_data['num_rules'] > 5:
            optimizations.append({
                "current": f"{input_data['num_rules']} rules",
                "recommendation": "Consolidate similar rules to reduce to 3-5 rules",
                "savings": (input_data['num_rules'] - 4) * 0.10,
                "complexity": "Medium"
            })
        
        if input_data['mqtt_messages_per_day'] > 10000:
            optimizations.append({
                "current": "Individual MQTT messages",
                "recommendation": "Use batch messages or MQTT topics to reduce message count",
                "savings": input_data['mqtt_messages_per_day'] * 0.00005,
                "complexity": "Low"
            })
        
        return optimizations