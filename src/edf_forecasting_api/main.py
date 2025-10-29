import os
import logging
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from contextlib import asynccontextmanager
from edf_forecasting_api.schema import InputData 
from edf_forecasting_api.model_manager import ModelManager
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# Create a manager model instance
model_manager = ModelManager(model_name="timeseries_xgboost_30min", check_interval=500)

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

@app.get("/")
def root():
    """Home Page"""
    return {"message": "Welcome to Consumption Forecasting API"}

@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return FileResponse("src/edf_forecasting_api/favicon.ico") if os.path.exists("src/edf_forecasting_api/favicon.ico") else JSONResponse(content={}, status_code=204)

@app.post("/predict")
def predict(data: InputData):
    consumptions = data.features
    n_predictions = data.n_predictions
    predictions = model_manager.predict(consumptions, n_predictions)
    return {"predictions": predictions}
