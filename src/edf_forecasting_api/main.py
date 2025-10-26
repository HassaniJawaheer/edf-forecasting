from fastapi import FastAPI
from edf_forecasting_api.schema import InputData
from edf_forecasting_api.model_utils import load_model, predict_consumptions

# Load model
model = load_model()

app = FastAPI(title="EDF Forecasting API")

@app.post("/predict")
def predict(data: InputData):
    consumptions = data.features
    n_predictions = data.n_predictions
    predictions = predict_consumptions(model, consumptions, n_predictions)
    return {"predictions": predictions}
