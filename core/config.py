import os
from pydantic_settings import BaseSettings  # <-- Changed import

class Settings(BaseSettings):
    app_name: str = "AWS Cost Predictor"
    model_path: str = "models/hybrid_model.pkl"
    data_path: str = "data/processed/aws_cost_data.csv"
    
    class Config:
        env_file = ".env"

settings = Settings()