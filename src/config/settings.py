import os
from dotenv import load_dotenv

# Load environment variables from .env file
dotenv_path = os.path.join(os.path.dirname(__file__), '../../.env')
load_dotenv(dotenv_path)

class Settings:
    APP_NAME: str = "AWS-IoT Cost Prediction"
    APP_VERSION: str = "1.0.0"
    # Paths - change to your actual paths
    AWS_DATA_PATH = os.getenv('AWS_DATA_PATH', 'D:/Projects/AWS-Iot-Cost-Prediction/data/raw/aws_billing_data.csv')
    IOT_DATA_PATH = os.getenv('IOT_DATA_PATH', 'D:/Projects/AWS-Iot-Cost-Prediction/data/iot/iot_costs.csv')
    
    AWS_MODEL_PATH = os.getenv('AWS_MODEL_PATH', 'D:/Projects/AWS-Iot-Cost-Prediction/models/aws_cost_model.h5')
    XGBOOST_MODEL_PATH = os.getenv('XGBOOST_MODEL_PATH', 'D:/Projects/AWS-Iot-Cost-Prediction/models/xgboost_model.json')
    IOT_MODEL_PATH = os.getenv('IOT_MODEL_PATH', 'D:/Projects/AWS-Iot-Cost-Prediction/models/iot_cost_model.h5')

settings = Settings()
