# src/schemas/aws_schema.py
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

class AWSInputItem(BaseModel):
    timestamp: datetime
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

class AWSInput(BaseModel):
    data: List[AWSInputItem]

class AWSPredictionOutput(BaseModel):
    timestamp: datetime
    predicted_cost: float
    cost_breakdown: dict
    optimization_suggestions: List[str]

class AWSBatchPredictionOutput(BaseModel):
    predictions: List[AWSPredictionOutput]
    overall_optimization_suggestions: List[str]