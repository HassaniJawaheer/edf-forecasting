from sklearn.model_selection import cross_val_score, KFold
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
import numpy as np

class Eco2mixCrossValidationXGBoostDay:
    def __init__(self, X, y, training_params, cv_params):
        self.X = X
        self.y = y
        self.training_params = training_params
        self.cv_params = cv_params

    def run(self):
        model = MultiOutputRegressor(XGBRegressor(**self.training_params))
        kf = KFold(**self.cv_params)

        scores = cross_val_score(
            model,
            self.X,
            self.y,
            cv=kf,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1
        )

        rmse_scores = -scores
        results = {
            "rmse_mean": float(np.mean(rmse_scores)),
            "rmse_std": float(np.std(rmse_scores)),
            "n_splits": self.cv_params.get("n_splits", 5),
            "shuffle": self.cv_params.get("shuffle", False)
        }

        return results
