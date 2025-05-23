from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

class IoTInputItem(BaseModel):
    timestamp: Optional[datetime] = None
    thing_name: str
    thing_type: str
    region: str
    attached_policies: int
    policy_names: Optional[str] = None
    attached_rules: int
    rule_names: Optional[str] = Field(None, alias="rules_names")  # Accepts both
    shadow_updates_per_day: int
    mqtt_messages_per_day: int
    http_requests_per_day: int
    device_connected_hours: float
    connection_type: str
    iot_data_transfer_mb: float = 0.0  # Default value

    class Config:
        allow_population_by_field_name = True

class IoTInput(BaseModel):
    data: List[IoTInputItem]

class IoTPredictionOutput(BaseModel):
    thing_name: str
    predicted_cost: float
    cost_breakdown: dict
    optimization_suggestions: List[str]

class IoTBatchPredictionOutput(BaseModel):
    predictions: List[IoTPredictionOutput]
    overall_optimization_suggestions: List[str]