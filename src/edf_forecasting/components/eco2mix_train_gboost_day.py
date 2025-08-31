import numpy as np
import logging
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score, root_mean_squared_error

logging.basicConfig(level=logging.INFO)

class Eco2mixTrainGBoostDay:
    def __init__(self, X_train, y_train, params):
        self.X = X_train
        self.y = y_train
        self.params = params

    def run(self):
        model = MultiOutputRegressor(XGBRegressor(**self.params))
        model.fit(self.X, self.y)

        y_pred = model.predict(self.X)
        scores = {
            "r2_score": float(r2_score(self.y, y_pred)),
            "rmse": float(root_mean_squared_error(self.y, y_pred))
        }

        metadata = {
            "model": "XGBoost_MultiOutput",
            "params_used": self.params,
            "n_samples": len(self.X)
        }

        return model, scores, metadata
