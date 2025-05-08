from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import pandas as pd
from datetime import datetime
from typing import List

router = APIRouter(prefix="/predict", tags=["predictions"])

class PredictionInput(BaseModel):
    date: str
    ec2_usage: float
    s3_usage: float
    lambda_usage: float
    rds_usage: float

class PredictionOutput(BaseModel):
    date: str
    predicted_cost: float
    confidence: float

@router.post("/cost", response_model=PredictionOutput)
async def predict_cost(input: PredictionInput):
    try:
        # Convert input to DataFrame
        input_dict = input.dict()
        df = pd.DataFrame([input_dict])
        
        # Make prediction
        model = router.app.state.model
        prediction = model.predict(df)[0]
        
        # For demo, using fixed confidence
        return {
            "date": input.date,
            "predicted_cost": round(prediction, 2),
            "confidence": 0.95  # In real app, calculate confidence
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/batch-cost", response_model=List[PredictionOutput])
async def predict_batch_cost(inputs: List[PredictionInput]):
    try:
        # Convert inputs to DataFrame
        input_dicts = [i.dict() for i in inputs]
        df = pd.DataFrame(input_dicts)
        
        # Make predictions
        model = router.app.state.model
        predictions = model.predict(df)
        
        # Prepare response
        results = []
        for i, row in enumerate(inputs):
            results.append({
                "date": row.date,
                "predicted_cost": round(predictions[i], 2),
                "confidence": 0.95
            })
        
        return results
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))