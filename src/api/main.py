# src/api/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routers import aws, iot
from src.config.settings import settings

app = FastAPI(
    title="AWS Cost Prediction For Iot Core and All Services API",
    description="API for predicting AWS costs and providing optimization recommendations",
    version="1.0.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(aws.router, prefix="/api/v1/aws", tags=["AWS Cost"])
app.include_router(iot.router, prefix="/api/v1/iot", tags=["IoT Core"])

@app.get("/")
async def root():
    return {"message": "AWS Cost Prediction For Iot Core and All Services API"}