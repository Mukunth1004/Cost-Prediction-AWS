# src/api/routers/aws.py
from fastapi import APIRouter, Depends, HTTPException
from typing import List
from datetime import date
from src.schemas import AWSCostPredictionInput, AWSCostPredictionOutput
from src.services.aws_cost import AWSCostService

router = APIRouter()

@router.post("/predict", response_model=AWSCostPredictionOutput)
async def predict_aws_cost(input_data: AWSCostPredictionInput):
    try:
        # Validate historical data length
        for service, data in input_data.historical_data.items():
            if len(data) < 30:
                raise HTTPException(
                    status_code=400,
                    detail=f"Need at least 30 days of historical data for {service}. Got {len(data)}"
                )
        
        service = AWSCostService()
        prediction = service.predict_cost(
            services=input_data.services,
            start_date=input_data.start_date,
            end_date=input_data.end_date,
            historical_data=input_data.historical_data
        )
        return prediction
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
@router.get("/services")
async def get_aws_services():
    """
    Get list of available AWS services for prediction
    """
    service = AWSCostService()
    return {"services": service.get_available_services()}