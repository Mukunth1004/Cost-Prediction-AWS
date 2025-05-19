import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from datetime import datetime
import os
from src.config import settings

class DataProcessor:
    def __init__(self):
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.region_encoder_aws = OneHotEncoder(handle_unknown='ignore', sparse=False)
        self.instance_type_encoder_aws = OneHotEncoder(handle_unknown='ignore', sparse=False)
        
        self.region_encoder_iot = OneHotEncoder(handle_unknown='ignore', sparse=False)
        self.thing_type_encoder_iot = OneHotEncoder(handle_unknown='ignore', sparse=False)
        self.connection_type_encoder_iot = OneHotEncoder(handle_unknown='ignore', sparse=False)
        
    def load_aws_data(self, file_path):
        """Load and preprocess AWS billing data with new columns"""
        df = pd.read_csv(file_path)
        df.fillna(0, inplace=True)
        
        # Use the specific encoders for AWS data
        region_encoded = self.region_encoder_aws.fit_transform(df[['region']])
        instance_type_encoded = self.instance_type_encoder_aws.fit_transform(df[['ec2_instance_type']])
        
        # Create feature DataFrame
        features = pd.DataFrame({
            'ec2_instance_hours': df['ec2_instance_hours'],
            'ec2_data_transfer_gb': df['ec2_data_transfer_gb'],
            's3_storage_gb': df['s3_storage_gb'],
            's3_requests': df['s3_requests'],
            'lambda_invocations': df['lambda_invocations'],
            'lambda_duration_ms': df['lambda_duration_ms'],
            'cloudwatch_logs_gb': df['cloudwatch_logs_gb'],
            'dynamodb_read_units': df['dynamodb_read_units'],
            'dynamodb_write_units': df['dynamodb_write_units'],
            'api_gateway_requests': df['api_gateway_requests']
        })
        
        # Add encoded features
        for i in range(region_encoded.shape[1]):
            features[f'region_{i}'] = region_encoded[:, i]
            
        for i in range(instance_type_encoded.shape[1]):
            features[f'instance_type_{i}'] = instance_type_encoded[:, i]
        
        # Scale features
        scaled_data = self.scaler.fit_transform(features)
        
        return scaled_data, df['estimated_cost_usd'], features.columns.tolist()
    
    def load_iot_data(self, file_path):
        """Load and preprocess IoT Core data with new columns"""
        df = pd.read_csv(file_path)
        df.fillna(0, inplace=True)
        
        # Convert timestamp to datetime and drop invalid timestamps
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df.dropna(subset=['timestamp'], inplace=True)
        
        # Extract time features
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['hour_of_day'] = df['timestamp'].dt.hour
        
        # One-hot encode categorical features separately
        region_encoded = self.region_encoder_iot.fit_transform(df[['region']])
        thing_type_encoded = self.thing_type_encoder_iot.fit_transform(df[['thing_type']])
        connection_type_encoded = self.connection_type_encoder_iot.fit_transform(df[['connection_type']])
        
        # Number of policies and rules
        num_policies = df['attached_policies'].apply(
            lambda x: len(str(x).split(',')) if pd.notna(x) and str(x).strip() != '' else 0)
        num_rules = df['attached_rules'].apply(
            lambda x: len(str(x).split(',')) if pd.notna(x) and str(x).strip() != '' else 0)
        
        # Combine all encoded categorical features horizontally
        encoded_cols = np.hstack([region_encoded, thing_type_encoded, connection_type_encoded])
        
        # Create feature DataFrame
        features = pd.DataFrame({
            'shadow_updates_per_day': df['shadow_updates_per_day'],
            'mqtt_messages_per_day': df['mqtt_messages_per_day'],
            'http_requests_per_day': df['http_requests_per_day'],
            'device_connected_hours': df['device_connected_hours'],
            'iot_data_transfer_mb': df['iot_data_transfer_mb'],
            'num_policies': num_policies,
            'num_rules': num_rules,
            'day_of_week': df['day_of_week'],
            'hour_of_day': df['hour_of_day']
        })
        
        # Add all encoded categorical features
        for i in range(encoded_cols.shape[1]):
            features[f'cat_{i}'] = encoded_cols[:, i]
        
        # Scale features
        scaled_data = self.scaler.fit_transform(features)
        
        return scaled_data, df['estimated_cost_usd'], features.columns.tolist()
    
    def create_time_series_dataset(self, data, target, n_steps):
        """Create time-series dataset for LSTM"""
        X, y = [], []
        for i in range(len(data) - n_steps):
            X.append(data[i:i + n_steps])
            y.append(target[i + n_steps])
        return np.array(X), np.array(y)
