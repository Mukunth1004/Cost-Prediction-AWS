# src/ml/training.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from .data_preprocessing import DataPreprocessor
from .feature_engineering import FeatureEngineer
from .models import HybridModel
import joblib
import os

class AWSCostTrainer:
    def __init__(self, config):
        self.config = config
        self.preprocessor = DataPreprocessor()
        self.feature_engineer = FeatureEngineer()
        self.model = None
        
    def load_data(self, data_path):
        df = pd.read_csv(data_path)
        return df
        
    def prepare_data(self, df):
    # Preprocess
        df = self.preprocessor.preprocess_aws_data(df)
    
    # Feature engineering
        df = self.feature_engineer.create_time_series_features(df)

        ts_features = [col for col in df.columns if col.startswith(('lag_', 'rolling_', 'ema_'))]
        if len(ts_features) != 16:
            raise ValueError(f"Expected 16 time-series features, got {len(ts_features)}")
    

    # Debugging: print time-series feature info
        time_series_features = [col for col in df.columns if col.startswith(('lag_', 'rolling_', 'ema_'))]
        print(f"\nTime-series features detected: {len(time_series_features)}")
        print(f"Sample features: {time_series_features[:5]}...")

        n_features = len(time_series_features)

    # Initialize model with correct feature dimensions
        self.model = HybridModel(n_features=n_features)

    # Split into sequence and tabular features
        services = [col for col in df.columns if col.startswith('Service_')]
        lag_features = [col for col in df.columns if col.startswith('lag_')]
        rolling_features = [col for col in df.columns if col.startswith('rolling_')]
        ema_features = [col for col in df.columns if col.startswith('ema_')]

        tabular_features = services + ['Day', 'Month', 'Year', 'DayOfWeek', 'DayOfYear', 'WeekOfYear']
        time_series_features = lag_features + rolling_features + ema_features

    # Create sequences
        X_seq, X_tab, y = self._create_sequences(
            df, 
            time_series_features, 
            tabular_features, 
            target='Cost',
            sequence_length=self.model.time_steps
       )

        return X_seq, X_tab, y

        
    def _create_sequences(self, df, ts_features, tab_features, target, sequence_length):
        X_seq, X_tab, y = [], [], []
        
        for service in df['Service'].unique():
            service_df = df[df['Service'] == service].sort_values('Date')

            if len(service_df) < sequence_length:
                padding = pd.DataFrame(0, index=range(sequence_length - len(service_df)), 
                          columns=service_df.columns)
                service_df = pd.concat([padding, service_df])
            for i in range(len(service_df) - sequence_length):
                # Time-series sequence
                seq = service_df.iloc[i:i+sequence_length][ts_features].values
                X_seq.append(seq)
                
                # Tabular features (from the last point in sequence)
                tab = service_df.iloc[i+sequence_length][tab_features].values
                X_tab.append(tab)
                
                # Target (next point after sequence)
                if i + sequence_length + 1 < len(service_df):
                    y.append(service_df.iloc[i+sequence_length+1][target])
                else:
                    # If no next point, use current as target (for last sequence)
                    y.append(service_df.iloc[i+sequence_length][target])
                    
        return np.array(X_seq), np.array(X_tab), np.array(y)
        
    def train(self, data_path):
        # Load and prepare data
        df = self.load_data(data_path)
        X_seq, X_tab, y = self.prepare_data(df)
        
        # Train-test split
        X_seq_train, X_seq_test, X_tab_train, X_tab_test, y_train, y_test = train_test_split(
            X_seq, X_tab, y, test_size=0.2, random_state=42)
            
        # Train model
        mae, rmse = self.model.train_hybrid(
            X_seq_train, X_tab_train, y_train,
            X_seq_test, X_tab_test, y_test,
            epochs=self.config['epochs'],
            batch_size=self.config['batch_size']
        )
        
        # Save model
        self.save_model()
        
        return mae, rmse
        
    def save_model(self):
        os.makedirs(self.config['model_dir'], exist_ok=True)
        joblib.dump(self.model, os.path.join(self.config['model_dir'], 'hybrid_model.pkl'))
        print(f"Model saved to {self.config['model_dir']}")