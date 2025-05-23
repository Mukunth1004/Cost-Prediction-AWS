import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib
from datetime import datetime
from src.config import config
from src.utils.logger import get_logger

logger = get_logger(__name__)

class IoTDataPreprocessor:
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.features = [
            "thing_name","thing_type", "region", "attached_policies", "attached_rules",
            "shadow_updates_per_day", "mqtt_messages_per_day", 
            "http_requests_per_day", "device_connected_hours", "connection_type","iot_data_transfer_mb"
        ]
        self.target = "estimated_cost_usd"
        self.feature_columns = None
        
    def load_data(self, file_path):
        """Load data with robust validation"""
        try:
            logger.info(f"Loading AWS data from {file_path}")
            df = pd.read_csv(file_path)

            required = self.features + [self.target]
            missing = set(required) - set(df.columns)
            if missing:
                raise ValueError(f"Missing required columns: {missing}")
                

            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df['hour'] = df['timestamp'].dt.hour.astype(float)
                df['day_of_week'] = df['timestamp'].dt.dayofweek.astype(float)
                df['month'] = df['timestamp'].dt.month.astype(float)
                df = df.drop(columns=['timestamp'])

            if 'policy_names' in df.columns:
                df['unique_policies'] = df['policy_names'].apply(lambda x: len(str(x).split(','))).astype(np.float32)
            if 'rule_names' in df.columns:
                df['unique_rules'] = df['rule_names'].apply(lambda x: len(str(x).split(','))).astype(np.float32)

            return df
        
        except Exception as e:
            logger.error(f"Failed to load data: {str(e)}")
            raise

    def load_data_from_dict(self, data_dict):
        try:
        # Convert input to DataFrame
            df = pd.DataFrame([data_dict])
            numeric_fields = [
            'attached_policies', 'attached_rules',
            'shadow_updates_per_day', 'mqtt_messages_per_day',
            'http_requests_per_day', 'device_connected_hours',
            'iot_data_transfer_mb'
            ]
            for field in numeric_fields:
                if field not in df.columns:
                    df[field] = 0
                df[field] = pd.to_numeric(df[field], errors='coerce').fillna(0)
            
            df['attached_policies'] = df.get('attached_policies', 0)
            df['attached_rules'] = df.get('attached_rules', 0)

            # Handle data transfer (accept both spellings)
            df['iot_data_transfer_mb'] = df.get('iot_data_transfer_mb', df.get('iot_data_transfer_mb', 0.0))
        
        # Ensure required columns exist
            required_cols = [
            "thing_type", "region", "attached_policies", "attached_rules",
            "shadow_updates_per_day", "mqtt_messages_per_day", 
            "http_requests_per_day", "device_connected_hours", 
            "connection_type", "iot_data_transfer_mb"
           ]
        
        # Add missing columns with default values
            for col in required_cols:
                if col not in df.columns:
                    if col in ['attached_policies', 'attached_rules']:
                        df[col] = 0  # Default count
                    elif col in ['shadow_updates_per_day', 'mqtt_messages_per_day', 
                           'http_requests_per_day', 'device_connected_hours',
                           'iot_data_transfer_mb']:
                        df[col] = 0.0  # Default numeric value
                    else:
                        df[col] = 'unknown'  # Default for categorical

        # Process policy and rule counts
            df['attached_policies'] = df['attached_policies'].apply(lambda x: len(str(x).split(',')) if pd.notnull(x) else 0).astype(np.float32)
            df['attached_rules'] = df['attached_rules'].apply(lambda x: len(str(x).split(',')) if pd.notnull(x) else 0).astype(np.float32)

        # Load feature columns used in training
            if not self.feature_columns:
                self.feature_columns = joblib.load(f"{config.SCALER_SAVE_PATH}/iot_feature_columns.pkl")
        
        # One-hot encode categorical variables
            categorical_cols = ['thing_type', 'region', 'connection_type']
            df = pd.get_dummies(df, columns=[col for col in categorical_cols if col in df.columns])
        
        # Ensure we have all expected columns
            missing_cols = set(self.feature_columns) - set(df.columns)
            for col in missing_cols:
                df[col] = 0
            
        # Remove any extra columns
            df = df[self.feature_columns]
        
        # Scale the data
            self.scaler = joblib.load(f"{config.SCALER_SAVE_PATH}/iot_scaler.pkl")
            X_scaled = self.scaler.transform(df.values)
        
            return X_scaled
        
        except Exception as e:
            logger.error(f"Failed to load data from dict: {str(e)}")
            raise

    def prepare_data(self, df, is_training=True):
        """Handle IoT data preprocessing consistently"""
        try:
            if df is None or df.empty:
                raise ValueError("Empty DataFrame received")
        
        # Convert policy and rule counts to numerical values
            df['attached_policies'] = df['attached_policies'].apply(lambda x: len(str(x).split(',')) if pd.notnull(x) else 0).astype(np.float32)
            df['attached_rules'] = df['attached_rules'].apply(lambda x: len(str(x).split(',')) if pd.notnull(x) else 0).astype(np.float32)

        # Drop non-numeric columns that shouldn't be used as features
            cols_to_drop = ['thing_name', 'policy_names', 'rule_names']
            df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])

        # One-hot encode categorical variables
            categorical_cols = ['thing_type', 'region', 'connection_type']
            df = pd.get_dummies(df, columns=[col for col in categorical_cols if col in df.columns])

        # Ensure consistent columns
            if is_training:
                self.feature_columns = [c for c in df.columns if c != self.target]
                joblib.dump(self.feature_columns, f"{config.SCALER_SAVE_PATH}/iot_feature_columns.pkl")
            else:
            # Ensure we have the same columns as training
                missing_cols = set(self.feature_columns) - set(df.columns)
                for col in missing_cols:
                    df[col] = 0
                extra_cols = set(df.columns) - set(self.feature_columns + [self.target])
                df = df.drop(columns=extra_cols)

        # Separate features and target
            X = df.drop(columns=[self.target]).values
            y = df[self.target].values

        # Scale features
            if is_training:
                X_scaled = self.scaler.fit_transform(X)
                joblib.dump(self.scaler, f"{config.SCALER_SAVE_PATH}/iot_scaler.pkl")
            else:
                X_scaled = self.scaler.transform(X)

            return X_scaled, y

        except Exception as e:
            logger.error(f"Data preparation failed: {str(e)}")
            raise

    def train_test_split(self, X, y, test_size=0.2, random_state=42):
        return train_test_split(X, y, test_size=test_size, random_state=random_state)

    def create_sequences(self, data, targets, sequence_length=24):
        X, y = [], []
        for i in range(len(data) - sequence_length):
            X.append(data[i:i+sequence_length])
            y.append(targets[i+sequence_length])
        return np.array(X), np.array(y)
