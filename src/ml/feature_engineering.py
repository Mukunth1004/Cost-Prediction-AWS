# src/ml/feature_engineering.py
import pandas as pd
import numpy as np

class FeatureEngineer:
    def create_time_series_features(self, df):
        
        for lag in [1, 3, 7, 14, 30]:
            df[f'lag_{lag}'] = df.groupby('Service')['Cost'].shift(lag)
            
        # Rolling statistics
        windows = [3, 7, 14, 30]
        for window in windows:
            df[f'rolling_mean_{window}'] = df.groupby('Service')['Cost'].transform(
                lambda x: x.rolling(window=window).mean())
            df[f'rolling_std_{window}'] = df.groupby('Service')['Cost'].transform(
                lambda x: x.rolling(window=window).std())
                
        # Exponential moving averages
        for alpha in [0.1, 0.3, 0.5]:
            df[f'ema_{alpha}'] = df.groupby('Service')['Cost'].transform(
                lambda x: x.ewm(alpha=alpha).mean())
                
        # Drop rows with NaN values from lag feature
        return df.dropna()
    
    def create_iot_features(self, df):
        # Interaction features
        df['RulePolicyInteraction'] = df['RuleComplexity'] * df['PolicyComplexity']
        
        # Cost per message
        df['CostPerMessage'] = df['TotalCost'] / (df['MessageCount'] + 1e-6)
        
        return df