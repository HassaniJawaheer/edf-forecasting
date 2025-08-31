import logging
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, root_mean_squared_error

logging.basicConfig(level=logging.INFO)

class Eco2mixTrainGBoost30min:
    def __init__(self, df_train, training_params, windows_size, target_col):
        self.df_train = df_train
        self.training_params = training_params
        self.windows_size = windows_size
        self.target_col = target_col
        self.X_train = None
        self.y_train = None
    
    def _create_windows(self):
        if self.target_col not in self.df_train.columns:
            raise ValueError(f"Target column '{self.target_col}' not found in train dataframe.")
        
        values = self.df_train[self.target_col].values

        if len(values) <= self.windows_size:
            raise ValueError("Insufficient data to create at least one windows.")
        
        self.X_train = np.lib.stride_tricks.sliding_window_view(values, window_shape=self.windows_size)[:-1]
        self.y_train = values[self.windows_size:]

    def run(self):
        self._create_windows()
        
        model = XGBRegressor(**self.training_params)
        model.fit(self.X_train, self.y_train)

        y_pred = model.predict(self.X_train)
        scores = {
            "r2_score": float(r2_score(self.y_train, y_pred)),
            "rmse": float(root_mean_squared_error(self.y_train, y_pred))
        }

        metadata = {
            "model": "XGBoostRegressor",
            "params_used": self.training_params,
            "n_samples": len(self.X_train)
        }

        return model, scores, metadata
