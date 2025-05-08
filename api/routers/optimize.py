from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import pandas as pd
from typing import List

router = APIRouter(prefix="/optimize", tags=["optimizations"])

class OptimizationInput(BaseModel):
    current_ec2_usage: float
    current_s3_usage: float
    current_lambda_usage: float
    current_rds_usage: float
    budget: float

class OptimizationRecommendation(BaseModel):
    service: str
    current_usage: float
    recommended_usage: float
    estimated_savings: float
    action: str

@router.post("/recommendations", response_model=List[OptimizationRecommendation])
async def get_optimization_recommendations(input: OptimizationInput):
    try:
        # This is a simplified version - in a real app, you'd use more sophisticated logic
        recommendations = []
        
        # EC2 recommendation
        if input.current_ec2_usage > 50:
            recommendations.append({
                "service": "EC2",
                "current_usage": input.current_ec2_usage,
                "recommended_usage": input.current_ec2_usage * 0.9,
                "estimated_savings": input.current_ec2_usage * 0.1 * 0.05,  # Example calculation
                "action": "Consider using reserved instances or scaling down during off-peak hours"
            })
        
        # S3 recommendation
        if input.current_s3_usage > 100:
            recommendations.append({
                "service": "S3",
                "current_usage": input.current_s3_usage,
                "recommended_usage": input.current_s3_usage * 0.85,
                "estimated_savings": input.current_s3_usage * 0.15 * 0.023,  # Example calculation
                "action": "Consider transitioning infrequently accessed data to S3 Infrequent Access"
            })
        
        # Lambda recommendation
        if input.current_lambda_usage > 200:
            recommendations.append({
                "service": "Lambda",
                "current_usage": input.current_lambda_usage,
                "recommended_usage": input.current_lambda_usage * 0.8,
                "estimated_savings": input.current_lambda_usage * 0.2 * 0.0000166667,  # Example calculation
                "action": "Optimize function memory allocation and execution time"
            })
        
        # RDS recommendation
        if input.current_rds_usage > 75:
            recommendations.append({
                "service": "RDS",
                "current_usage": input.current_rds_usage,
                "recommended_usage": input.current_rds_usage * 0.85,
                "estimated_savings": input.current_rds_usage * 0.15 * 0.025,  # Example calculation
                "action": "Consider using Aurora Serverless or right-sizing instances"
            })
        
        # Filter based on budget if provided
        if input.budget > 0:
            total_current = (input.current_ec2_usage + input.current_s3_usage + 
                           input.current_lambda_usage + input.current_rds_usage)
            if total_current > input.budget:
                # Apply more aggressive recommendations
                for rec in recommendations:
                    rec['recommended_usage'] *= 0.9
                    rec['estimated_savings'] *= 1.1
                    rec['action'] = "URGENT: " + rec['action']
        
        return recommendations
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))