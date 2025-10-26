import mlflow
from typing import List
import numpy as np

def load_model():

    # Model path
    logged_model = 'runs:/0de33fa0279642a1b74d039e6536bc6e/model'

    # Load model as a PyFuncModel.
    loaded_model = mlflow.pyfunc.load_model(logged_model)
    return loaded_model

def predict_consumptions(model, consumptions: List, n_predictions: int):
    X = np.array(consumptions)
    predictions = []
    for _ in range(n_predictions):
        y_pred = model.predict(X)
        predictions.append(y_pred)
        X = np.hstack([X[:, 1:], y_pred.reshape(-1, 1)])
    return predictions

    