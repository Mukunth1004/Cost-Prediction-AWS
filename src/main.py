from fastapi import FastAPI
from fastapi import HTTPException
from fastapi.middleware.cors import CORSMiddleware
from src.schemas.aws_schema import AWSInput, AWSBatchPredictionOutput
from src.schemas.iot_schema import IoTInput, IoTBatchPredictionOutput
from src.services.aws_cost_services import AWSCostService
from src.services.iot_cost_services import IoTCostService
from src.utils.logger import get_logger
import uvicorn

logger = get_logger(__name__)

app = FastAPI(
    title="AWS & IoT Cost Prediction API",
    description="API for predicting AWS service costs and IoT Core thing costs with optimization recommendations",
    version="1.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

aws_service = AWSCostService()
iot_service = IoTCostService()

@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    try:
        aws_service.load_model()
        iot_service.load_model()
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        raise

@app.get("/")
def read_root():
    return {"message": "AWS & IoT Cost Prediction API"}

@app.post("/predict/aws", response_model=AWSBatchPredictionOutput)
def predict_aws_cost(input_data: AWSInput):
    """Predict AWS service costs"""
    try:
        return aws_service.predict_cost(input_data)
    except Exception as e:
        logger.error(f"Error predicting AWS costs: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict/iot", response_model=IoTBatchPredictionOutput)
def predict_iot_cost(input_data: IoTInput):
    """Predict IoT Core thing costs"""
    try:
        return iot_service.predict_cost(input_data)
    except Exception as e:
        logger.error(f"Error predicting IoT costs: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/train/aws")
def train_aws_model():
    """Train the AWS cost prediction model"""
    try:
        return aws_service.train_model()
    except Exception as e:
        logger.error(f"Error training AWS model: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/train/iot")
def train_iot_model():
    """Train the IoT cost prediction model"""
    try:
        return iot_service.train_model()
    except Exception as e:
        logger.error(f"Error training IoT model: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

