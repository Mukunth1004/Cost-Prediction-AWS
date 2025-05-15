# src/api/routers/iot.py
from fastapi import APIRouter, HTTPException
from typing import List
from src.schemas import IoTCostPredictionInput, IoTCostPredictionOutput, IoTOptimizationRecommendation
from src.services.iot_cost import IoTCostService

router = APIRouter()

@router.post("/predict", response_model=IoTCostPredictionOutput)
async def predict_iot_cost(input_data: IoTCostPredictionInput):
    """
    Predict IoT Core costs based on configuration
    """
    try:
        service = IoTCostService()
        prediction = service.predict_cost(
            message_volume=input_data.message_volume,
            rules=input_data.rules,
            policies=input_data.policies
        )
        return prediction
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/optimize", response_model=List[IoTOptimizationRecommendation])
async def optimize_iot_cost(input_data: IoTCostPredictionInput):
    """
    Get optimization recommendations for IoT Core configuration
    """
    try:
        service = IoTCostService()
        recommendations = service.get_optimization_recommendations(
            message_volume=input_data.message_volume,
            rules=input_data.rules,
            policies=input_data.policies
        )
        return recommendations
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))