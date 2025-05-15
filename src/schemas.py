# src/api/schemas.py
from pydantic import BaseModel
from typing import List, Dict, Optional
from datetime import date

class AWSCostPredictionInput(BaseModel):
    services: List[str]
    start_date: date
    end_date: date
    historical_data: Optional[Dict[str, List[float]]] = None

class AWSCostPredictionOutput(BaseModel):
    predictions: Dict[str, List[float]]  # service -> list of daily predictions
    total_cost: float
    cost_breakdown: Dict[str, float]  # service -> percentage of total

class IoTRule(BaseModel):
    name: str
    sql: str
    actions: List[str]
    enabled: bool
    error_action: Optional[str] = None

class IoTPolicy(BaseModel):
    name: str
    statements: List[Dict]  # Simplified policy statements

class IoTCostPredictionInput(BaseModel):
    message_volume: int  # Messages per day
    rules: List[IoTRule]
    policies: List[IoTPolicy]

class IoTCostPredictionOutput(BaseModel):
    message_cost: float
    rule_execution_cost: float
    policy_evaluation_cost: float
    total_cost: float
    cost_breakdown: Dict[str, float]  # cost component -> percentage

class IoTOptimizationRecommendation(BaseModel):
    area: str  # "rules", "policies", "message_processing"
    current_cost: float
    potential_savings: float
    recommendation: str
    implementation: str