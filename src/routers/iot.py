from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from datetime import datetime
from typing import List
from src.services.iot_cost import IoTCostService
from src.schemas.iot_cost import (
    IoTCostPredictionInput,
    IoTCostPredictionOutput,
    IoTCostComponent,
    IoTOptimization
)

router = APIRouter()
iot_service = IoTCostService()

@router.post("/predict", response_model=IoTCostPredictionOutput)
async def predict_iot_cost(input_data: IoTCostPredictionInput):
    """Predict IoT costs with rule/policy breakdown"""
    try:
        prediction = service.predict_cost(input_data)
        components = service.get_cost_components(input_data)
        optimizations = service.get_optimizations(input_data)
        
        return {
            "prediction": prediction,
            "cost_components": components,
            "optimizations": optimizations
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/rule-costs")
async def get_rule_costs():
    """Get cost information for different rule types"""
    return {
        "basic_rule": 0.05,
        "sql_rule": 0.10,
        "lambda_rule": 0.15,
        "republish_rule": 0.08
    }

@router.get("/policy-costs")
async def get_policy_costs():
    """Get cost information for different policy types"""
    return {
        "basic_policy": 0.03,
        "certificate_policy": 0.05,
        "advanced_policy": 0.08
    }