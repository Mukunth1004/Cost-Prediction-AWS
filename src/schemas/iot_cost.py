from pydantic import BaseModel
from datetime import datetime
from typing import List, Optional

class IoTCostPredictionInput(BaseModel):
    thing_name: str
    thing_type: str
    region: str
    attached_policies: str
    attached_rules: str
    shadow_updates_per_day: int
    mqtt_messages_per_day: int
    http_requests_per_day: int
    device_connected_hours: float
    connection_type: str
    iot_data_transfer_mb: float
    historical_data: Optional[List[dict]] = None

class IoTCostComponent(BaseModel):
    component: str
    count: float
    cost: float
    explanation: str

class IoTOptimization(BaseModel):
    current: str
    recommendation: str
    savings: float
    complexity: str

class IoTCostPredictionOutput(BaseModel):
    prediction: float
    cost_components: List[IoTCostComponent]
    optimizations: List[IoTOptimization]