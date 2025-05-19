import pandas as pd
from datetime import datetime
from typing import List, Dict, Any
from src.models.prediction import IoTCostPredictor
from src.utils.iot_utils import (
    calculate_rule_costs,
    calculate_policy_costs,
    generate_iot_optimizations
)
from src.config import settings
from src.schemas.iot_cost import (
    IoTCostPredictionInput,
    IoTCostComponent,
    IoTOptimization
)

class IoTCostService:
    def __init__(self):
        self.predictor = IoTCostPredictor()
    
    def predict_cost(self, input_data: IoTCostPredictionInput) -> float:
        """Predict IoT costs with explanations"""
        # Convert input to DataFrame
        df = self._create_input_dataframe(input_data)
        
        # Get prediction
        prediction = self.predictor.predict(df)
        return prediction
    
    def get_cost_components(self, input_data: IoTCostPredictionInput) -> List[IoTCostComponent]:
        """Generate detailed cost breakdown with rules and policies"""
        df = self._create_input_dataframe(input_data)
        
        # Get rule and policy costs
        rules = input_data.attached_rules.split(',') if input_data.attached_rules else []
        policies = input_data.attached_policies.split(',') if input_data.attached_policies else []
        
        rule_costs = calculate_rule_costs(rules)
        policy_costs = calculate_policy_costs(policies)
        
        # Create components list
        components = []
        
        # Add rule costs
        for rule in rule_costs:
            components.append(IoTCostComponent(
                component=f"Rule: {rule['rule_name']}",
                count=1,
                cost=rule['cost'],
                explanation=f"{rule['type']} at ${rule['cost']:.2f} per rule"
            ))
        
        # Add policy costs
        for policy in policy_costs:
            components.append(IoTCostComponent(
                component=f"Policy: {policy['policy_name']}",
                count=1,
                cost=policy['cost'],
                explanation=f"{policy['type']} at ${policy['cost']:.2f} per policy"
            ))
        
        # Add message costs
        message_cost = input_data.mqtt_messages_per_day * 0.0001
        components.append(IoTCostComponent(
            component="MQTT Messages",
            count=input_data.mqtt_messages_per_day,
            cost=message_cost,
            explanation=f"{input_data.mqtt_messages_per_day} messages at $0.0001 per message"
        ))
        
        return components
    
    def get_optimizations(self, input_data: IoTCostPredictionInput) -> List[IoTOptimization]:
        """Generate IoT-specific optimizations"""
        rules = input_data.attached_rules.split(',') if input_data.attached_rules else []
        policies = input_data.attached_policies.split(',') if input_data.attached_policies else []
        
        rule_costs = calculate_rule_costs(rules)
        policy_costs = calculate_policy_costs(policies)
        
        optimizations = generate_iot_optimizations(
            rule_costs,
            policy_costs,
            input_data.mqtt_messages_per_day
        )
        
        return [IoTOptimization(**opt) for opt in optimizations]
    
    def _create_input_dataframe(self, input_data: IoTCostPredictionInput) -> pd.DataFrame:
        """Create input DataFrame from prediction input"""
        data = {
            "timestamp": [datetime.now()],
            "thing_name": [input_data.thing_name],
            "thing_type": [input_data.thing_type],
            "region": [input_data.region],
            "attached_policies": [input_data.attached_policies],
            "attached_rules": [input_data.attached_rules],
            "shadow_updates_per_day": [input_data.shadow_updates_per_day],
            "mqtt_messages_per_day": [input_data.mqtt_messages_per_day],
            "http_requests_per_day": [input_data.http_requests_per_day],
            "device_connected_hours": [input_data.device_connected_hours],
            "connection_type": [input_data.connection_type],
            "iot_data_transfer_mb": [input_data.iot_data_transfer_mb],
            "estimated_cost_usd": [0]  # Placeholder
        }
        
        if input_data.historical_data:
            hist_df = pd.DataFrame(input_data.historical_data)
            return pd.concat([hist_df, pd.DataFrame(data)], ignore_index=True)
        
        return pd.DataFrame(data)