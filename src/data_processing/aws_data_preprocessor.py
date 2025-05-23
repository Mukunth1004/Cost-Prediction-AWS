# src/data_processing/aws_data_preprocessor.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib
from datetime import datetime
from src.config import config
from src.utils.logger import get_logger

logger = get_logger(__name__)

class AWSDataPreprocessor:
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.features = [
            "region",
            "ec2_instance_hours",
            "ec2_instance_type",
            "ec2_data_transfer_gb",
            "s3_storage_gb",
            "s3_requests",
            "lambda_invocations",
            "lambda_duration_ms",
            "cloudwatch_logs_gb",
            "dynamodb_read_units",
            "dynamodb_write_units",
            "api_gateway_requests"
        ]
        self.target = "estimated_cost_usd"
        self.feature_columns = None

    def load_data(self, file_path):
        """Load data with robust validation"""
        try:
            logger.info(f"Loading AWS data from {file_path}")
            df = pd.read_csv(file_path)
        
        # Validate required columns
            required_cols = self.features + [self.target]
            missing_cols = set(required_cols) - set(df.columns)
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Convert timestamp if exists
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df['hour'] = df['timestamp'].dt.hour.astype(float)
                df['day_of_week'] = df['timestamp'].dt.dayofweek.astype(float)
                df['month'] = df['timestamp'].dt.month.astype(float)
                df = df.drop(columns=['timestamp'])
            return df
        
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
        
    def load_data_from_dict(self, data_dict):
        """Convert input dictionary to processed DataFrame"""
        try:
            # Handle both list of dicts and single dict input
            if isinstance(data_dict, dict):
                df = pd.DataFrame([data_dict])
            else:
                df = pd.DataFrame(data_dict)

            if not self.feature_columns:
                self.feature_columns = joblib.load(f"{config.SCALER_SAVE_PATH}/aws_feature_columns.pkl")
        
            # Ensure all features exist
            categorical_cols = ['region', 'ec2_instance_type']
            df = pd.get_dummies(df, columns=[col for col in categorical_cols if col in df.columns])

            missing_cols = set(self.feature_columns) - set(df.columns)
            for col in missing_cols:
                df[col] = 0 

            extra_cols = set(df.columns) - set(self.feature_columns)
            df = df.drop(columns=extra_cols, errors='ignore')

            df = df[self.feature_columns]
    
            df = df.apply(pd.to_numeric, errors='coerce').fillna(0)
            
            self.scaler = joblib.load(f"{config.SCALER_SAVE_PATH}/aws_scaler.pkl")
            return self.scaler.transform(df.values)

        except Exception as e:
            logger.error(f"Failed to process input data: {str(e)}")
            raise ValueError(f"Feature processing failed: {str(e)}")
    
    def prepare_data(self, df, is_training=True):
        """Properly handle categorical features and ensure numeric conversion"""
        try:
        # 1. Handle timestamp if exists
            if df is None or df.empty:
                raise ValueError("Empty DataFrame received")
        
        # 2. Convert categorical columns to dummy variables
            categorical_cols = ['region', 'ec2_instance_type']
            df = pd.get_dummies(df, columns=[col for col in categorical_cols if col in df.columns])
        
        # 3. Store feature columns during training
            if is_training:
                self.feature_columns = [col for col in df.columns if col != self.target]
                joblib.dump(self.feature_columns, f"{config.SCALER_SAVE_PATH}/aws_feature_columns.pkl")
                logger.info(f"Saved feature columns: {len(self.feature_columns)} features")

            if not is_training and self.feature_columns:
                missing = set(self.feature_columns) - set(df.columns)
                for col in missing:
                    df[col] = 0
                extra = set(df.columns) - set(self.feature_columns + [self.target])
                df = df.drop(columns=extra, errors='ignore')
                df = df[self.feature_columns + ([self.target] if self.target in df.columns else [])]
        
        # 5. Convert all data to numeric
            X = df.drop(columns=[self.target], errors='ignore').apply(pd.to_numeric, errors='coerce').fillna(0)
            y = df[self.target].values if self.target in df.columns else None
        
        # 6. Scale features
            if is_training:
                X_scaled = self.scaler.fit_transform(X)
                joblib.dump(self.scaler, f"{config.SCALER_SAVE_PATH}/aws_scaler.pkl")
            else:
                X_scaled = self.scaler.transform(X)
        
            return X_scaled, y
        
        except Exception as e:
            logger.error(f"Data preparation failed: {str(e)}")
            raise

    def train_test_split(self, X, y, test_size=0.2, random_state=42):
        """Split data into train and test sets"""
        if len(X) < 10:  # For very small datasets
            return X[:-1], X[-1:], y[:-1], y[-1:]
        return train_test_split(X, y, test_size=test_size, random_state=random_state)

    
    def create_sequences(self, data, targets, sequence_length=24):
        """Create time series sequences for LSTM"""
        X, y = [], []
        for i in range(len(data) - sequence_length):
            X.append(data[i:i+sequence_length])
            y.append(targets[i+sequence_length])
        return np.array(X), np.array(y)