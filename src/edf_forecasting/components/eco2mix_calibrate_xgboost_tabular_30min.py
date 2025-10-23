import numpy as np
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)

class Eco2mixCalibrateTabXGBoost30min:
    """Calibrate prediction intervals for an XGBoost model using calibration data."""

    def __init__(
        self,
        df_cal,
        model,
        error_type="raw",
        target_col="Consommation",
        drop_duplicates=True,
        ensure_float32=True
    ):
        self.model = model
        self.df_cal = df_cal
        self.error_type = error_type
        self.target_col = target_col
        self.drop_duplicates = drop_duplicates
        self.ensure_float32 = ensure_float32

        self.q_inf = None
        self.q_sup = None
        self.X_cal = None
        self.y_cal = None

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

        # Convert object columns to category for XGBoost
        cat_cols = X.select_dtypes(include=["object"]).columns
        for col in cat_cols:
            X[col] = X[col].astype("category")

        # Convert numerical columns to float32 for performance
        if self.ensure_float32:
            num_cols = X.select_dtypes(include=["int", "float"]).columns
            X[num_cols] = X[num_cols].astype("float32")

        logging.info(
            f"Preprocessing complete. Shape X: {X.shape}, y: {y.shape}. "
            f"Categorical cols: {len(cat_cols)}, numeric cols: {len(X.columns) - len(cat_cols)}."
        )

        return X, y

    def run(self, alpha=0.05):
        """Compute lower and upper quantile errors for prediction interval calibration."""
        self.X_cal, self.y_cal = self._preprocess_data(self.df_cal)

        y_pred = self.model.predict(self.X_cal)
        errors = self.y_cal - y_pred

        if self.error_type == "absolute":
            errors = np.abs(errors)

        self.q_inf = np.quantile(errors, alpha / 2)
        self.q_sup = np.quantile(errors, 1 - alpha / 2)

        logging.info(
            f"Calibration complete. q_inf={self.q_inf:.5f}, q_sup={self.q_sup:.5f}, alpha={alpha}"
        )

        return float(self.q_inf), float(self.q_sup)
