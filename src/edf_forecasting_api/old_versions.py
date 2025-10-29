import mlflow
from typing import List
import numpy as np

def load_model():

    # Model path
    #logged_model = 'runs:/0de33fa0279642a1b74d039e6536bc6e/model'
    logged_model = "models:/timeseries_xgboost_30min/Production"

    # Load model as a PyFuncModel.
    loaded_model = mlflow.pyfunc.load_model(logged_model)
    return loaded_model

def predict_consumptions(model, consumptions: list, n_predictions: int):

    X = np.array(consumptions)
    predictions = []

    for _ in range(n_predictions):
        y_pred = model.predict(X)

        y_pred_list = y_pred.tolist() if hasattr(y_pred, "tolist") else y_pred
        predictions.append(y_pred_list)
        X = np.hstack([X[:, 1:], np.array(y_pred).reshape(-1, 1)])

    return [float(p) for sublist in predictions for p in np.ravel(sublist)]


import os
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from edf_forecasting_api.schema import InputData
from edf_forecasting_api.old_versions import load_model, predict_consumptions

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

    