import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, MultiHeadAttention
from tensorflow.keras.layers import LayerNormalization, Concatenate
from tensorflow.keras.callbacks import EarlyStopping
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from src.models.data_processing import DataProcessor
from src.config.settings import settings


class HybridModelTrainer:
    def __init__(self, n_steps=10, n_head=4, key_dim=64):
        self.n_steps = n_steps
        self.n_head = n_head
        self.key_dim = key_dim
        self.processor = DataProcessor()
    
    def build_glstm_attention_model(self, n_features):
        """Build gLSTM with multi-head attention model"""
        inputs = Input(shape=(self.n_steps, n_features))
        
        # gLSTM layers
        lstm1 = LSTM(128, return_sequences=True, activation='gelu')(inputs)
        lstm1 = Dropout(0.2)(lstm1)
        lstm2 = LSTM(64, return_sequences=True, activation='gelu')(lstm1)
        lstm2 = Dropout(0.2)(lstm2)
        
        # Multi-head attention
        attention_output = MultiHeadAttention(
            num_heads=self.n_head, 
            key_dim=self.key_dim)(lstm2, lstm2)
        
        # Skip connection
        attention_output = Concatenate()([lstm2, attention_output])
        attention_output = LayerNormalization()(attention_output)
        
        # Final layers
        flatten = tf.keras.layers.Flatten()(attention_output)
        dense1 = Dense(64, activation='gelu')(flatten)
        outputs = Dense(1)(dense1)
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer='adam', loss='mse')
        
        return model
    
    def train_aws_model(self, data_path=settings.AWS_DATA_PATH):
        """Train model for AWS cost prediction"""
        print(f"Loading AWS data from {data_path}")
        X, y, features = self.processor.load_aws_data(data_path)
        n_features = len(features)
        
        X_ts, y_ts = self.processor.create_time_series_dataset(X, y, self.n_steps)
        
        split = int(0.8 * len(X_ts))
        X_train, X_test = X_ts[:split], X_ts[split:]
        y_train, y_test = y_ts[:split], y_ts[split:]
        
        lstm_model = self.build_glstm_attention_model(n_features)
        early_stop = EarlyStopping(monitor='val_loss', patience=10)
        
        lstm_model.fit(
            X_train, y_train,
            epochs=100,
            batch_size=32,
            validation_data=(X_test, y_test),
            callbacks=[early_stop],
            verbose=1
        )
        
        lstm_model.save(settings.AWS_MODEL_PATH)
        
        xgb_model = XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9
        )
        
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        X_test_flat = X_test.reshape(X_test.shape[0], -1)
        
        xgb_model.fit(X_train_flat, y_train)
        xgb_model.save_model(settings.XGBOOST_MODEL_PATH)
        
        lstm_preds = lstm_model.predict(X_test).flatten()
        xgb_preds = xgb_model.predict(X_test_flat)
        final_preds = (lstm_preds + xgb_preds) / 2
        
        mae = mean_absolute_error(y_test, final_preds)
        rmse = np.sqrt(mean_squared_error(y_test, final_preds))
        
        print(f"AWS Model - MAE: {mae:.2f}, RMSE: {rmse:.2f}")
        
        return lstm_model, xgb_model
    
    def train_iot_model(self, data_path=settings.IOT_DATA_PATH):
        """Train model for IoT cost prediction"""
        print(f"Loading IoT data from {data_path}")
        X, y, features = self.processor.load_iot_data(data_path)
        n_features = len(features)
        
        X_ts, y_ts = self.processor.create_time_series_dataset(X, y, self.n_steps)
        
        split = int(0.8 * len(X_ts))
        X_train, X_test = X_ts[:split], X_ts[split:]
        y_train, y_test = y_ts[:split], y_ts[split:]
        
        lstm_model = self.build_glstm_attention_model(n_features)
        early_stop = EarlyStopping(monitor='val_loss', patience=10)
        
        lstm_model.fit(
            X_train, y_train,
            epochs=100,
            batch_size=32,
            validation_data=(X_test, y_test),
            callbacks=[early_stop],
            verbose=1
        )
        
        lstm_model.save(settings.IOT_MODEL_PATH)
        
        xgb_model = XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9
        )
        
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        X_test_flat = X_test.reshape(X_test.shape[0], -1)
        
        xgb_model.fit(X_train_flat, y_train)
        
        lstm_preds = lstm_model.predict(X_test).flatten()
        xgb_preds = xgb_model.predict(X_test_flat)
        final_preds = (lstm_preds + xgb_preds) / 2
        
        mae = mean_absolute_error(y_test, final_preds)
        rmse = np.sqrt(mean_squared_error(y_test, final_preds))
        
        print(f"IoT Model - MAE: {mae:.2f}, RMSE: {rmse:.2f}")
        
        return lstm_model, xgb_model


if __name__ == "__main__":
    trainer = HybridModelTrainer()
    print("Training AWS model...")
    trainer.train_aws_model()
    print("Training IoT model...")
    trainer.train_iot_model()
