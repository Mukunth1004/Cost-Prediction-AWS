# tests/test_aws_prediction.py
import pytest
from fastapi.testclient import TestClient
from src.main import app
from src.schemas.aws_schema import AWSInputItem, AWSInput
from datetime import datetime

client = TestClient(app)

def test_predict_aws_cost():
    # Test data matching the training data format
    test_data = [
        AWSInputItem(
            timestamp=datetime.now(),
            region="us-east-1",
            ec2_instance_hours=720,
            ec2_instance_type="t3.medium",
            ec2_data_transfer_gb=100,
            s3_storage_gb=500,
            s3_requests=10000,
            lambda_invocations=1000000,
            lambda_duration_ms=3000,
            cloudwatch_logs_gb=50,
            dynamodb_read_units=100,
            dynamodb_write_units=50,
            api_gateway_requests=100000
        )
    ]
    
    response = client.post(
        "/predict/aws",
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