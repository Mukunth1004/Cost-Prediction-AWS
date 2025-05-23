# src/config.py
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    AWS_DATA_PATH = os.getenv('AWS_DATA_PATH', 'data/aws_cost_data.csv')
    IOT_DATA_PATH = os.getenv('IOT_DATA_PATH', 'data/iot_cost_data.csv')
    MODEL_SAVE_PATH = os.getenv('MODEL_SAVE_PATH', 'models/')
    SCALER_SAVE_PATH = os.getenv('SCALER_SAVE_PATH', 'models/scalers/')
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')

config = Config()