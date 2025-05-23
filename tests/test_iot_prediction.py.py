# tests/test_iot_prediction.py
import pytest
from fastapi.testclient import TestClient
from src.main import app
from src.schemas.iot_schema import IoTInputItem, IoTInput
from datetime import datetime

client = TestClient(app)

def test_predict_iot_cost():
    # Test data matching the training data format
    test_data = [
        IoTInputItem(
            timestamp=datetime.now(),
            thing_name="test-device-1",
            thing_type="temperature-sensor",
            region="us-east-1",
            attached_policies=2,
            policy_names="policy1,policy2",
            attached_rules=3,
            rules_names="rule1,rule2,rule3",
            shadow_updates_per_day=1000,
            mqtt_messages_per_day=5000,
            http_requests_per_day=100,
            device_connected_hours=24,
            connection_type="wifi"
        )
    ]
    
    response = client.post(
        "/predict/iot",
        json={"data": [item.dict() for item in test_data]}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "predictions" in data
    assert len(data["predictions"]) == len(test_data)
    assert "predicted_cost" in data["predictions"][0]
    assert "cost_breakdown" in data["predictions"][0]
    assert "optimization_suggestions" in data["predictions"][0]
    assert "overall_optimization_suggestions" in data