from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from datetime import date
from typing import List
from src.services.aws_cost import AWSCostService
from src.schemas.aws_cost import (
    AWSCostPredictionInput,
    AWSCostPredictionOutput,
    CostBreakdownItem,
    OptimizationRecommendation
)

router = APIRouter()
aws_service = AWSCostService()

@router.post("/predict", response_model=AWSCostPredictionOutput)
async def predict_aws_cost(input_data: AWSCostPredictionInput):
    """Predict AWS costs with detailed breakdown"""
    try:
        prediction = service.predict_cost(input_data)
        breakdown = service.get_cost_breakdown(input_data)
        optimizations = service.get_optimizations(input_data)
        
        return {
            "prediction": prediction,
            "cost_breakdown": breakdown,
            "optimization_recommendations": optimizations
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/historical")
async def get_historical_costs(start_date: date, end_date: date):
    """Get historical AWS cost data"""
    try:
        return service.get_historical_data(start_date, end_date)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))