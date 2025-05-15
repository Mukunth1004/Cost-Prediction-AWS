# src/ml/data_preprocessing.py
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

class DataPreprocessor:
    def __init__(self):
        self.cost_scaler = MinMaxScaler()
        self.iot_scaler = MinMaxScaler()
        self.preprocessor = self._build_preprocessor()
        
    def _build_preprocessor(self):
        """Build feature preprocessing pipeline"""
        numeric_features = ['IoT_Things', 'IoT_Rules', 'IoT_Policies']
        categorical_features = ['Service']
        
        return ColumnTransformer(
            transformers=[
                ('num', Pipeline([
                    ('scaler', MinMaxScaler())
                ]), numeric_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ])
    
    def preprocess_aws_data(self, df):
        """Full preprocessing pipeline for AWS data (renamed from preprocess_data)"""
        # Convert date and extract features
        df['Date'] = pd.to_datetime(df['Date'])
        df['Day'] = df['Date'].dt.day
        df['Month'] = df['Date'].dt.month
        df['Year'] = df['Date'].dt.year
        df['DayOfWeek'] = df['Date'].dt.dayofweek
        df['DayOfYear'] = df['Date'].dt.dayofyear
        df['WeekOfYear'] = df['Date'].dt.isocalendar().week
        
        # One-hot encode services
        services = pd.get_dummies(df['Service'], prefix='Service')
        df = pd.concat([df, services], axis=1)
        
        # Normalize cost
        df['Cost'] = self.cost_scaler.fit_transform(df[['Cost']])
        
        return df
    
    def inverse_transform_cost(self, scaled_cost):
        """Convert scaled cost back to USD"""
        return self.cost_scaler.inverse_transform(scaled_cost.reshape(-1, 1))