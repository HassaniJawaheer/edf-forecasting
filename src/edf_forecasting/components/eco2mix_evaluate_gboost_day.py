import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.base import BaseEstimator

class Eco2mixEvaluateGBoostDay:
    def __init__(self, model: BaseEstimator):
        self.model = model
    def run(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        scores = {
            "r2_score": float(r2),
            "rmse": float(rmse)
        }

        return scores

