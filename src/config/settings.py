# src/config/settings.py
from pydantic import BaseSettings
import os
from pathlib import Path

class Settings(BaseSettings):
    APP_NAME: str = "AWS Cost Predictor"
    DEBUG: bool = False
    MODEL_PATH: str = str(Path(__file__).parent.parent.parent / "models/hybrid_model.pkl")
    ALLOWED_ORIGINS: list = ["http://localhost:3000", "http://127.0.0.1:3000"]
    
    class Config:
        env_file = ".env"

settings = Settings()