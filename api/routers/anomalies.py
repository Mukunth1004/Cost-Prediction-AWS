from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import pandas as pd
from typing import List

router = APIRouter(prefix="/anomalies", tags=["anomalies"])

class AnomalyDetectionInput(BaseModel):
    date: str
    cost: float
    ec2_usage: float
    s3_usage: float
    lambda_usage: float
    rds_usage: float

class AnomalyDetectionResult(BaseModel):
    date: str
    is_anomaly: bool
    confidence: float
    explanation: str

@router.post("/detect", response_model=AnomalyDetectionResult)
async def detect_anomaly(input: AnomalyDetectionInput):
    try:
        # Convert input to DataFrame
        input_dict = input.dict()
        df = pd.DataFrame([input_dict])
        
        # Detect anomaly
        model = router.app.state.model
        anomaly_score = model.detect_anomalies(df)[0]
        
        is_anomaly = anomaly_score == -1
        confidence = abs(anomaly_score)  # Convert to confidence measure
        
        explanation = "Normal usage pattern"
        if is_anomaly:
            explanation = "Anomalous pattern detected - "
            if input.cost > (input.ec2_usage + input.s3_usage + input.lambda_usage + input.rds_usage) * 1.5:
                explanation += "Cost is significantly higher than expected given usage patterns"
            elif input.ec2_usage > input_dict.get('historical_ec2_avg', input.ec2_usage * 2):
                explanation += "EC2 usage is significantly higher than historical average"
            else:
                explanation += "Unusual spending pattern detected"
        
        return {
            "date": input.date,
            "is_anomaly": is_anomaly,
            "confidence": round(confidence, 2),
            "explanation": explanation
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/batch-detect", response_model=List[AnomalyDetectionResult])
async def detect_batch_anomalies(inputs: List[AnomalyDetectionInput]):
    try:
        # Convert inputs to DataFrame
        input_dicts = [i.dict() for i in inputs]
        df = pd.DataFrame(input_dicts)
        
        # Detect anomalies
        model = router.app.state.model
        anomaly_scores = model.detect_anomalies(df)
        
        # Prepare response
        results = []
        for i, row in enumerate(inputs):
            is_anomaly = anomaly_scores[i] == -1
            confidence = abs(anomaly_scores[i])
            
            explanation = "Normal usage pattern"
            if is_anomaly:
                explanation = "Anomalous pattern detected"
            
            results.append({
                "date": row.date,
                "is_anomaly": is_anomaly,
                "confidence": round(confidence, 2),
                "explanation": explanation
            })
        
        return results
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))