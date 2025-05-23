# src/services/aws_cost_service.py
import numpy as np
import pandas as pd
import joblib
from fastapi import HTTPException  # Add at top of file
from src.models.aws_model import AWSModel
from src.data_processing.aws_data_preprocessor import AWSDataPreprocessor
from src.services.optimization_service import OptimizationService
from src.config import config
from src.utils.logger import get_logger

logger = get_logger(__name__)

class AWSCostService:
    def __init__(self):
        self.model = AWSModel()
        self.preprocessor = AWSDataPreprocessor()
        self.is_trained = False
    
    def train_model(self):
        """Updated training method"""
        try:
        # Load and preprocess data
            df = self.preprocessor.load_data(config.AWS_DATA_PATH)
            if df is None:
                raise ValueError("No data loaded")
            X, y = self.preprocessor.prepare_data(df, is_training=True)
            if X is None or y is None:
                raise ValueError("Data preparation failed")
        
        # Train-test split
            from sklearn.model_selection import train_test_split
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
        
        # Train model
            result = self.model.train(X_train, y_train, X_val, y_val)
            self.is_trained = True
            self.model.save_model()

            if result['status'] != 'success':
                raise ValueError(result['message'])

            return {
            'status': 'success',
            'metrics': result.get('metrics', {}),
            'feature_importances': result.get('feature_importances', [])
            }
        
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            return {'status': 'error', 'message': str(e)}
    
    def load_model(self):
        """Load the trained model"""
        try:
            self.model.load_model()
            self.scaler = joblib.load(f"{config.SCALER_SAVE_PATH}/aws_scaler.pkl")
            self.is_trained = True
            logger.info("AWS cost prediction model loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            self.is_trained = False
            return False
    
    # src/services/aws_cost_service.py
    def predict_cost(self, input_data):
        """Handle prediction with proper error responses"""
        try:
            if not self.is_trained and not self.load_model():
                raise HTTPException(status_code=400, detail="Model not loaded")

            input_dicts = [item.dict() for item in input_data.data]
            try:
                df = self.preprocessor.load_data_from_dict(input_dicts)
            except ValueError as e:
                logger.error(f"Feature processing error: {str(e)}")
                raise HTTPException(status_code=400, detail=str(e))

            if df.shape[1] != len(self.preprocessor.feature_columns):
                error_msg = (
                f"Feature mismatch. Expected {len(self.preprocessor.feature_columns)} features, "
                f"got {df.shape[1]}. Missing/extra features may cause this."
                )
                logger.error(error_msg)
                raise HTTPException(status_code=400, detail=error_msg)

            predictions = self.model.predict(df)

            prediction_details = []
            all_cost_breakdowns = []

             
            for item, pred in zip(input_data.data, predictions):
                breakdown = self._generate_cost_breakdown(item)
                suggestions = OptimizationService.optimize_aws_resources(breakdown, item.dict())
                
                prediction_details.append({
                'timestamp': item.timestamp,
                'predicted_cost': float(pred),
                'cost_breakdown': breakdown,
                'optimization_suggestions': suggestions
                })
                all_cost_breakdowns.append(breakdown)

            return {
            'predictions': prediction_details,
            'overall_optimization_suggestions': OptimizationService.generate_overall_optimization_suggestions(all_cost_breakdowns),
            'feature_count': len(self.preprocessor.feature_columns)

            }

        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            raise HTTPException(status_code=400, detail=str(e))
    
    def _generate_cost_breakdown(self, item):
        """Generate cost breakdown for a single AWS service item"""
        # This is a simplified version - in a real app, you'd use AWS pricing API
        breakdown = {
            'ec2': item.ec2_instance_hours * 0.05,  # Example rate
            's3': item.s3_storage_gb * 0.023 + item.s3_requests * 0.0004,
            'lambda': (item.lambda_invocations * 0.0000002) + 
                     (item.lambda_duration_ms * 0.0000166667),
            'data_transfer': item.ec2_data_transfer_gb * 0.09,
            'cloudwatch': item.cloudwatch_logs_gb * 0.5,
            'dynamodb': (item.dynamodb_read_units * 0.00013) + 
                       (item.dynamodb_write_units * 0.00065),
            'api_gateway': item.api_gateway_requests * 0.000001
        }
        
        # Calculate percentages
        total = sum(breakdown.values())
        if total > 0:
            breakdown = {k: {'cost': v, 'percentage': (v / total) * 100} 
                        for k, v in breakdown.items()}
        
        return breakdown
    
    def _generate_optimization_suggestions(self, item, predicted_cost):
        """Generate optimization suggestions for a single AWS service item"""
        suggestions = []
        
        # EC2 suggestions
        if item.ec2_instance_hours > 720:  # More than 30 days
            suggestions.append("Consider Reserved Instances for EC2 to save up to 75%")
        
        # S3 suggestions
        if item.s3_storage_gb > 500:
            suggestions.append("Consider S3 Intelligent-Tiering for storage over 500GB")
        
        # Lambda suggestions
        if item.lambda_duration_ms > 3000:
            suggestions.append("Optimize Lambda functions to reduce duration below 3s")
        
        return suggestions
    
    def _generate_overall_suggestions(self, input_data, predictions):
        """Generate overall optimization suggestions"""
        suggestions = []
        total_cost = sum(predictions)

        if total_cost > 1000:
            suggestions.append("Your monthly AWS cost is projected to exceed $1000. Consider AWS Savings Plans.")

        ec2_hours = sum(item.ec2_instance_hours for item in input_data.data)
        if ec2_hours > 0 and (total_cost / ec2_hours) > 0.1:
            suggestions.append("High EC2 cost per hour detected. Consider downsizing instances.")

        return suggestions
