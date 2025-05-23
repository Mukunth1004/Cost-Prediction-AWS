from fastapi.testclient import TestClient
from src.main import app
import pytest

client = TestClient(app)

def test_predict_iot_cost():
    test_data = {
        "thing_name": "test_device_1",
        "thing_type": "temperature_sensor",
        "region": "us-west-2",
        "attached_policies": "basic_policy,certificate_policy",
        "attached_rules": "basic_rule,sql_rule",
        "shadow_updates_per_day": 100,
        "mqtt_messages_per_day": 5000,
        "http_requests_per_day": 200,
        "device_connected_hours": 24,
        "connection_type": "mqtt",
        "iot_data_transfer_mb": 50
    }
    
    response = client.post("/iot/predict", json=test_data)
    assert response.status_code == 200
    assert isinstance(response.json()["prediction"], float)
    assert any(c['component'] == 'Rules' for c in response.json()["cost_components"])
    assert any(c['component'] == 'Policies' for c in response.json()["cost_components"])

def test_rule_costs():
    response = client.get("/iot/rule-costs")
    assert response.status_code == 200
    assert "basic_rule" in response.json()

def test_policy_costs():
    response = client.get("/iot/policy-costs")
    assert response.status_code == 200
    assert "basic_policy" in response.json()