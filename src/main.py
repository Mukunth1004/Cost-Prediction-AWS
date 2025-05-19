from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.routers import aws, iot
from src.config.settings import settings

app = FastAPI(
    title=settings.APP_NAME,
    description="API for predicting AWS and IoT Core costs with optimization recommendations",
    version=settings.APP_VERSION
)

# CORS settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(aws.router, prefix="/aws", tags=["AWS Cost Prediction"])
app.include_router(iot.router, prefix="/iot", tags=["IoT Cost Prediction"])

@app.get("/")
def read_root():
    return {"message": "AWS-IoT Cost Prediction API"}