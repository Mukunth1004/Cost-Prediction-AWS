from pydantic import BaseModel
from datetime import date
from typing import List, Optional

class AWSCostPredictionInput(BaseModel):
    region: str
    ec2_instance_hours: float
    ec2_instance_type: str
    ec2_data_transfer_gb: float
    s3_storage_gb: float
    s3_requests: int
    lambda_invocations: int
    lambda_duration_ms: float
    cloudwatch_logs_gb: float
    dynamodb_read_units: int
    dynamodb_write_units: int
    api_gateway_requests: int
    historical_data: Optional[List[dict]] = None

class CostBreakdownItem(BaseModel):
    service: str
    usage: float
    cost: float
    explanation: str

class OptimizationRecommendation(BaseModel):
    current_setup: str
    recommended_setup: str
    potential_savings: float
    implementation_complexity: str

class AWSCostPredictionOutput(BaseModel):
    prediction: float
    cost_breakdown: List[CostBreakdownItem]
    optimization_recommendations: List[OptimizationRecommendation]