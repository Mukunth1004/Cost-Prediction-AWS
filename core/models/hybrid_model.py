import numpy as np
import pandas as pd
from tensorflow import keras
from xgboost import XGBRegressor
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from joblib import dump, load
import os

class HybridCostPredictor:
    def __init__(self):
        self.tft_model = None
        self.xgb_model = None
        self.anomaly_detector = None
        self.scaler = MinMaxScaler()
        self.features = None
        self.target = 'cost'

    def preprocess_data(self, data):
        """Preprocess the AWS cost data"""
        data['date'] = pd.to_datetime(data['date'])
        data['day_of_week'] = data['date'].dt.dayofweek
        data['day_of_month'] = data['date'].dt.day
        data['month'] = data['date'].dt.month
        data['year'] = data['date'].dt.year

        self.features = [
            'day_of_week', 'day_of_month', 'month', 'year',
            'ec2_usage', 's3_usage', 'lambda_usage', 'rds_usage'
        ]

        X = data[self.features]
        y = data[self.target]

        X_scaled = self.scaler.fit_transform(X)

        return X_scaled, y.values, X.columns.tolist()

    def build_tft_model(self, input_shape):
        inputs = keras.layers.Input(shape=input_shape)

        lstm1 = keras.layers.LSTM(64, return_sequences=True)(inputs)
        lstm1 = keras.layers.Dropout(0.2)(lstm1)

        lstm2 = keras.layers.LSTM(32, return_sequences=True)(lstm1)
        lstm2 = keras.layers.Dropout(0.2)(lstm2)

        attention = keras.layers.MultiHeadAttention(
            num_heads=4, key_dim=32)(lstm2, lstm2)

        merged = keras.layers.Concatenate()([lstm2, attention])

        pooled = keras.layers.GlobalAveragePooling1D()(merged)
        dense = keras.layers.Dense(64, activation='relu')(pooled)
        dense = keras.layers.Dropout(0.1)(dense)
        output = keras.layers.Dense(1)(dense)

        model = keras.models.Model(inputs=inputs, outputs=output)
        model.compile(optimizer='adam', loss='mse')

        return model

    def train(self, data):
        """Train the hybrid model"""
        X, y, features = self.preprocess_data(data)
        self.features = features

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        X_train_reshaped = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
        X_test_reshaped = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

        self.tft_model = self.build_tft_model((1, X_train.shape[1]))
        self.tft_model.fit(
            X_train_reshaped, y_train,
            epochs=50,
            batch_size=32,
            validation_data=(X_test_reshaped, y_test),
            verbose=1  # corrected from 'verbose='
        )

        self.xgb_model = XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.9,
            colsample_bytree=0.9
        )
        self.xgb_model.fit(X_train, y_train)

        self.anomaly_detector = IsolationForest(contamination=0.05)
        self.anomaly_detector.fit(X_train)

        return self

    def predict(self, X):
        """Make predictions using hybrid approach"""
        if isinstance(X, pd.DataFrame):
            X = X[self.features]

        X_scaled = self.scaler.transform(X)
        X_reshaped = X_scaled.reshape(X_scaled.shape[0], 1, X_scaled.shape[1])

        tft_pred = self.tft_model.predict(X_reshaped).flatten()
        xgb_pred = self.xgb_model.predict(X_scaled)

        hybrid_pred = (tft_pred + xgb_pred) / 2
        return hybrid_pred

    def detect_anomalies(self, X):
        """Detect anomalous cost patterns"""
        if isinstance(X, pd.DataFrame):
            X = X[self.features]

        X_scaled = self.scaler.transform(X)
        anomalies = self.anomaly_detector.predict(X_scaled)
        return anomalies

    def save(self, path):
        """Save the model to disk"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Save Keras model with .keras extension
        tft_model_path = os.path.splitext(path)[0] + "_tft.keras"
        self.tft_model.save(tft_model_path)
    
    # Save other components with joblib
        dump({
        'xgb_model': self.xgb_model,
        'anomaly_detector': self.anomaly_detector,
        'scaler': self.scaler,
        'features': self.features,
        'target': self.target,
        'tft_model_path': tft_model_path  # Store path to Keras model
        }, path)
    
    @classmethod
    def load(cls, path):
        """Load the model from disk"""
        data = load(path)
    
        model = cls()
        model.xgb_model = data['xgb_model']
        model.anomaly_detector = data['anomaly_detector']
        model.scaler = data['scaler']
        model.features = data['features']
        model.target = data['target']
    
    # Load Keras model
        tft_model_path = data['tft_model_path']
        model.tft_model = keras.models.load_model(tft_model_path)
    
        return model