# src/models/xgboost_model.py
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import numpy as np
from src.config import config
from src.utils.logger import get_logger

logger = get_logger(__name__)

class XGBoostModel:
    def __init__(self):
        self.model = None
        self.scaler = None
        
    def train(self, X_train, y_train, X_val, y_val):
        """Train XGBoost model with early stopping"""
        try:
            self.model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                early_stopping_rounds=10,
                eval_metric='mae'
            )
            
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=10
            )
            
            # Evaluate performance
            val_pred = self.model.predict(X_val)
            mae = mean_absolute_error(y_val, val_pred)
            rmse = np.sqrt(mean_squared_error(y_val, val_pred))
            
            # Save model
            joblib.dump(self.model, f"{config.MODEL_SAVE_PATH}/aws_xgb_model.pkl")
            
            return {
                'status': 'success',
                'metrics': {'mae': mae, 'rmse': rmse},
                'feature_importances': self.model.feature_importances_.tolist()
            }
            
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            return {'status': 'error', 'message': str(e)}
            
    def predict(self, X):
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model not trained")
        return self.model.predict(X)
        
    def evaluate(self, X, y):
        """Evaluate model performance"""
        if self.model is None:
            raise ValueError("Model not trained")
        preds = self.predict(X)
        return {
            'mae': mean_absolute_error(y, preds),
            'rmse': np.sqrt(mean_squared_error(y, preds))
        }