from fastapi.testclient import TestClient
from src.main import app
from datetime import date, timedelta
import pytest

client = TestClient(app)

def test_predict_aws_cost():
    test_data = {
        "region": "us-east-1",
        "ec2_instance_hours": 720,
        "ec2_instance_type": "t3.medium",
        "ec2_data_transfer_gb": 50,
        "s3_storage_gb": 100,
        "s3_requests": 5000,
        "lambda_invocations": 10000,
        "lambda_duration_ms": 500000,
        "cloudwatch_logs_gb": 10,
        "dynamodb_read_units": 100,
        "dynamodb_write_units": 50,
        "api_gateway_requests": 2000
    }
    
    response = client.post("/aws/predict", json=test_data)
    assert response.status_code == 200
    assert isinstance(response.json()["prediction"], float)
    assert len(response.json()["cost_breakdown"]) > 0
    assert len(response.json()["optimization_recommendations"]) >= 0

def test_historical_data():
    start = date.today() - timedelta(days=30)
    end = date.today()
    
    response = client.get(f"/aws/historical?start_date={start}&end_date={end}")
    assert response.status_code == 200
    assert len(response.json()) > 0