import numpy as np
import pandas as pd
import logging
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

logging.basicConfig(level=logging.INFO)

class Eco2mixEvaluateTabXGBoost30min:
    """Evaluate an XGBoost model on 30-min interval data with prediction intervals."""

    def __init__(
        self,
        model,
        df_test,
        q_inf,
        q_sup,
        quantile,
        target_col,
        drop_duplicates=True,
        ensure_float32=True
    ):
        self.model = model
        self.df_test = df_test
        self.q_inf = q_inf
        self.q_sup = q_sup
        self.quantile = quantile
        self.target_col = target_col
        self.drop_duplicates = drop_duplicates
        self.ensure_float32 = ensure_float32

        self.X_test = None
        self.y_test = None
        self.y_pred = None
        self.lower = None
        self.upper = None

    def _preprocess_data(self, df: pd.DataFrame):
        df = df.copy()

        if self.drop_duplicates:
            df = df.drop_duplicates().reset_index(drop=True)

        if self.target_col not in df.columns:
            raise ValueError(f"Target column '{self.target_col}' not found in dataframe.")

        # Delete unused columns
        df = df.drop(columns=["tempo"])

        y = df[self.target_col]
        X = df.drop(columns=[self.target_col])

        # Convert object columns to category
        cat_cols = X.select_dtypes(include=["object"]).columns
        for col in cat_cols:
            X[col] = X[col].astype("category")

        # Convert numerical columns to float32
        if self.ensure_float32:
            num_cols = X.select_dtypes(include=["int", "float"]).columns
            X[num_cols] = X[num_cols].astype("float32")

        logging.info(
            f"Preprocessing complete. Shape X: {X.shape}, y: {y.shape}. "
            f"Categorical cols: {len(cat_cols)}, numeric cols: {len(X.columns) - len(cat_cols)}."
        )

        return X, y

    def _predict(self):
        self.X_test, self.y_test = self._preprocess_data(self.df_test)

        self.y_pred = self.model.predict(self.X_test)

        self.lower = self.y_pred + self.q_inf
        self.upper = self.y_pred + self.q_sup

    def _pinball_loss(self, y_true, y_pred, quantile):
        delta = y_true - y_pred
        return np.mean(np.maximum(quantile * delta, (quantile - 1) * delta))

    def run(self):
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
