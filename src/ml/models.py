# src/ml/models.py
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, MultiHeadAttention, LayerNormalization
from tensorflow.keras.regularizers import l2
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

class HybridModel:
    def __init__(self, time_steps=30, n_features=16, lstm_units=128, num_heads=4):
        self.time_steps = time_steps
        self.n_features = n_features
        self.lstm_units = lstm_units
        self.num_heads = num_heads
        self.lstm_model = self.build_lstm_attention_model()
        self.xgb_model = XGBRegressor(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_alpha=0.1,
            reg_lambda=0.1
        )
        
    def build_lstm_attention_model(self):
        # Input layer
        inputs = Input(shape=(None, self.n_features))
        
        # LSTM layers
        lstm_out = LSTM(self.lstm_units, return_sequences=True, 
                       kernel_regularizer=l2(0.01))(inputs)
        lstm_out = Dropout(0.2)(lstm_out)
        lstm_out = LSTM(self.lstm_units, return_sequences=True,
                       kernel_regularizer=l2(0.01))(lstm_out)
        
        # Multi-head attention
        attention_output = MultiHeadAttention(
            num_heads=self.num_heads, key_dim=self.lstm_units)(lstm_out, lstm_out)
        
        # Skip connection and layer normalization
        attention_output = LayerNormalization()(
            tf.keras.layers.add([lstm_out, attention_output]))
        
        # Global average pooling and dense layers
        gap = tf.keras.layers.GlobalAveragePooling1D()(attention_output)
        dense = Dense(64, activation='relu', kernel_regularizer=l2(0.01))(gap)
        dense = Dropout(0.2)(dense)
        output = Dense(1)(dense)
        
        model = Model(inputs=inputs, outputs=output)
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        return model
    
    def train_hybrid(self, X_train_seq, X_train_tab, y_train, 
                    X_val_seq=None, X_val_tab=None, y_val=None,
                    epochs=50, batch_size=32):
        # Train LSTM model
        print("Training LSTM with attention...")
        self.lstm_model.fit(
            X_train_seq, y_train,
            validation_data=(X_val_seq, y_val) if X_val_seq is not None else None,
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )
        
        # Get LSTM predictions as features
        lstm_preds = self.lstm_model.predict(X_train_seq).flatten()
        
        # Combine with tabular features
        X_train_combined = np.column_stack([X_train_tab, lstm_preds])
        
        # Train XGBoost
        print("Training XGBoost...")
        self.xgb_model.fit(X_train_combined, y_train)
        
        # Validation if provided
        if X_val_seq is not None:
            val_lstm_preds = self.lstm_model.predict(X_val_seq).flatten()
            X_val_combined = np.column_stack([X_val_tab, val_lstm_preds])
            y_pred = self.xgb_model.predict(X_val_combined)
            
            mae = mean_absolute_error(y_val, y_pred)
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            
            print(f"Validation MAE: {mae:.2f}, RMSE: {rmse:.2f}")
            
            return mae, rmse
        return None, None
    
    def predict(self, X_seq, X_tab):
        lstm_preds = self.lstm_model.predict(X_seq).flatten()
        X_combined = np.column_stack([X_tab, lstm_preds])
        return self.xgb_model.predict(X_combined)