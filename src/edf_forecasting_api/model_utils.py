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
    import numpy as np

    X = np.array(consumptions)
    predictions = []

    for _ in range(n_predictions):
        y_pred = model.predict(X)

        y_pred_list = y_pred.tolist() if hasattr(y_pred, "tolist") else y_pred
        predictions.append(y_pred_list)
        X = np.hstack([X[:, 1:], np.array(y_pred).reshape(-1, 1)])

    return [float(p) for sublist in predictions for p in np.ravel(sublist)]

    