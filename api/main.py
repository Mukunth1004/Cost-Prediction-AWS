from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from core.config import settings
from core.models.hybrid_model import HybridCostPredictor
import pandas as pd
import joblib
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Add to main.py
app = FastAPI(title=settings.app_name)

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model at startup
@app.on_event("startup")
async def startup_event():
    if os.path.exists(settings.model_path):
        app.state.model = HybridCostPredictor.load(settings.model_path)
    else:
        # Initialize with empty model (in production, you'd train first)
        app.state.model = HybridCostPredictor()

# Include routers
from api.routers import predict, optimize, anomalies
app.include_router(predict.router)
app.include_router(optimize.router)
app.include_router(anomalies.router)

@app.get("/")
async def root():
    return {"message": "AWS Cost Prediction API"}