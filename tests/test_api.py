# tests/test_api.py
from fastapi.testclient import TestClient
from src.api.main import app
from datetime import date, timedelta

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "AWS Cost Prediction API"}

def test_aws_predict():
    test_data = {
        "services": ["AmazonEC2", "AmazonS3"],
        "start_date": str(date.today() + timedelta(days=1)),
        "end_date": str(date.today() + timedelta(days=7)),
    }
    response = client.post("/api/v1/aws/predict", json=test_data)
    assert response.status_code == 200
    data = response.json()
    assert "predictions" in data
    assert "total_cost" in data
    assert "cost_breakdown" in data
    assert len(data["predictions"]) == 2

def test_iot_predict():
    test_data = {
        "message_volume": 100000,
        "rules": [{
            "name": "test_rule",
            "sql": "SELECT * FROM 'test/topic'",
            "actions": ["s3"],
            "enabled": True
        }],
        "policies": [{
            "name": "test_policy",
            "statements": [{"Effect": "Allow", "Action": "iot:*", "Resource": "*"}]
        }]
    }
    response = client.post("/api/v1/iot/predict", json=test_data)
    assert response.status_code == 200
    data = response.json()
    assert "message_cost" in data
    assert "rule_execution_cost" in data
    assert "policy_evaluation_cost" in data
    assert "total_cost" in data