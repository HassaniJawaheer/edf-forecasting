import os
import logging
from uuid import uuid4
from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator
from fastapi.responses import FileResponse, JSONResponse
from contextlib import asynccontextmanager
from src.edf_forecasting_api.schema import InputData, FeedbackData
from src.edf_forecasting_api.model_manager import ModelManager
from src.edf_forecasting_api.logger_utils import log_feedback, log_predictions
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Variable name
MODEL_NAME = os.getenv("MODEL_NAME", "timeseries_xgboost_30min")
MODEL_CHECK_INTERVAL = int(os.getenv("MODEL_CHECK_INTERVAL", "300"))

# Create a manager model instance
model_manager = ModelManager(model_name=MODEL_NAME, check_interval=MODEL_CHECK_INTERVAL)

# FastAPI's lifespan context to handle startup and shutdown tasks
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    model_manager.load_model()
    model_manager.start_watcher()
    logging.info("Model monitoring enabled.")

    yield

    # Shutdown
    model_manager.stop_watcher()

# FastAPI app
app = FastAPI(title="Consumption Forecasting API", lifespan=lifespan)

# Expose Prometheus metrics for API monitoring
Instrumentator = Instrumentator().instrument(app=app)
Instrumentator.expose(app)

@app.get("/")
def root():
    """Home Page"""
    return {"message": "Welcome to Consumption Forecasting API"}

@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return FileResponse("src/edf_forecasting_api/favicon.ico") if os.path.exists("src/edf_forecasting_api/favicon.ico") else JSONResponse(content={}, status_code=204)

@app.post("/predict")
def predict(data: InputData):
    # Generate prediction identifier
    prediction_id = str(uuid4())

    # Exctract data
    consumptions = data.features
    n_predictions = data.n_predictions

    # Prediction
    predictions = model_manager.predict(consumptions, n_predictions)

    # Logging
    model_version = model_manager.current_version or 0 # 0 means unknown
    model_name = model_manager.model_name or "unknown"
    log_predictions(consumptions, predictions, model_name, model_version, n_predictions, prediction_id)

    return {"predictions": predictions, "prediction_id": prediction_id}

@app.post("/feedback")
def feedback(data: FeedbackData):
    log_feedback(data.inputs, data.outputs, data.prediction_id)
    return {"message": "Feedback saved"}