import pandas as pd
import numpy as np
from datetime import date, timedelta
from typing import List, Dict, Any
from src.models.prediction import AWSCostPredictor
from src.utils.aws_utils import generate_cost_breakdown, generate_recommendations
from src.config import settings
from src.schemas.aws_cost import (
    AWSCostPredictionInput,
    CostBreakdownItem,
    OptimizationRecommendation
)

class AWSCostService:
    def __init__(self):
        self.predictor = AWSCostPredictor()
    
    def predict_cost(self, input_data: AWSCostPredictionInput) -> float:
        """Predict AWS costs with explanations"""
        # Convert input to DataFrame
        df = self._create_input_dataframe(input_data)
        
        # Get prediction
        prediction = self.predictor.predict(df)
        return prediction
    
    def get_cost_breakdown(self, input_data: AWSCostPredictionInput) -> List[CostBreakdownItem]:
        """Generate detailed cost breakdown"""
        df = self._create_input_dataframe(input_data)
        breakdown_data = generate_cost_breakdown(df.iloc[-1])
        return [CostBreakdownItem(**item) for item in breakdown_data]
    
    def get_optimizations(self, input_data: AWSCostPredictionInput) -> List[OptimizationRecommendation]:
        """Generate optimization recommendations"""
        df = self._create_input_dataframe(input_data)
        optimizations = generate_recommendations(df.iloc[-1])
        return [OptimizationRecommendation(**opt) for opt in optimizations]
    
    def get_historical_data(self, start_date: date, end_date: date) -> List[Dict[str, Any]]:
        """Get historical cost data (mock implementation)"""
        # In a real implementation, this would query a database
        date_range = pd.date_range(start_date, end_date)
        data = {
            "date": date_range,
            "cost": np.random.uniform(100, 500, len(date_range))
        }
        return pd.DataFrame(data).to_dict(orient="records")
    
    def _create_input_dataframe(self, input_data: AWSCostPredictionInput) -> pd.DataFrame:
        """Create input DataFrame from prediction input"""
        # Create a single row DataFrame from the input data
        data = {
            "region": [input_data.region],
            "ec2_instance_hours": [input_data.ec2_instance_hours],
            "ec2_instance_type": [input_data.ec2_instance_type],
            "ec2_data_transfer_gb": [input_data.ec2_data_transfer_gb],
            "s3_storage_gb": [input_data.s3_storage_gb],
            "s3_requests": [input_data.s3_requests],
            "lambda_invocations": [input_data.lambda_invocations],
            "lambda_duration_ms": [input_data.lambda_duration_ms],
            "cloudwatch_logs_gb": [input_data.cloudwatch_logs_gb],
            "dynamodb_read_units": [input_data.dynamodb_read_units],
            "dynamodb_write_units": [input_data.dynamodb_write_units],
            "api_gateway_requests": [input_data.api_gateway_requests],
            "estimated_cost_usd": [0]  # Placeholder
        }
        
        # Add historical data if provided
        if input_data.historical_data:
            hist_df = pd.DataFrame(input_data.historical_data)
            return pd.concat([hist_df, pd.DataFrame(data)], ignore_index=True)
        
        return pd.DataFrame(data)