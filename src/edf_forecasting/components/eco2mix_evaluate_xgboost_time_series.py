import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

class XGBEvaluate30min:
    def __init__(self, model, df_test, q_inf, q_sup, quantile, windows_size, target_col):
        self.model = model
        self.df_test = df_test
        self.q_inf = q_inf
        self.q_sup = q_sup
        self.quantile = quantile
        self.windows_size = windows_size
        self.target_col = target_col

        self.X_test = None
        self.y_test = None
        self.y_pred = None
        self.lower = None
        self.upper = None

    def _create_windows(self):
        if self.target_col not in self.df_test.columns:
            raise ValueError(f"Target column '{self.target_col}' not found in test dataframe.")

        values = self.df_test[self.target_col].values

        if len(values) <= self.windows_size:
            raise ValueError("Insufficient data to create at least one window.")

        self.X_test = np.lib.stride_tricks.sliding_window_view(values, window_shape=self.windows_size)[:-1]
        self.y_test = values[self.windows_size:]

    def _predict(self):
        self._create_windows()

        self.y_pred = self.model.predict(self.X_test)
        self.lower = self.y_pred + self.q_inf
        self.upper = self.y_pred + self.q_sup

    def _pinball_loss(self, y_true, y_pred, quantile):
        delta = y_true - y_pred
        return np.mean(np.maximum(quantile * delta, (quantile - 1) * delta))

    def run(self):
        self._create_windows()
        self._predict()

        y_true = self.y_test
        y_pred = self.y_pred
        lower = self.lower
        upper = self.upper

        return {
            "rmse": float(root_mean_squared_error(y_true, y_pred)),
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "r2": float(r2_score(y_true, y_pred)),
            "coverage": float(np.mean((y_true >= lower) & (y_true <= upper))),
            "interval_width": float(np.mean(upper - lower)),
            "pinball_loss_lower": float(self._pinball_loss(y_true, lower, quantile=self.quantile / 2)),
            "pinball_loss_upper": float(self._pinball_loss(y_true, upper, quantile=1 - self.quantile / 2)),
            "overflow_rate": float(np.mean(y_true > upper))
        }
