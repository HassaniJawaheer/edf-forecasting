import os
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from edf_forecasting_api.schema import InputData
from edf_forecasting_api.model_utils import load_model, predict_consumptions

# Load model
model = load_model()

app = FastAPI(title="Consumption Forecasting API")

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
    predictions = predict_consumptions(model, consumptions, n_predictions)
    return {"predictions": predictions}
