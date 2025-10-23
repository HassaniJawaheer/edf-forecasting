import logging
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, root_mean_squared_error

logging.basicConfig(level=logging.INFO)

class Eco2mixTrainTabXGBoost30min:
    """Train an XGBoost regressor on 30-min interval tabular Eco2mix data."""

    def __init__(
        self,
        df_train,
        training_params,
        target_col="Consommation",
        drop_duplicates=True,
        ensure_float32=True
    ):
        self.df_train = df_train
        self.training_params = training_params
        self.target_col = target_col
        self.drop_duplicates = drop_duplicates
        self.ensure_float32 = ensure_float32

        self.X_train = None
        self.y_train = None

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

        # Convert numerical columns to float32
        if self.ensure_float32:
            num_cols = X.select_dtypes(include=["int", "float"]).columns
            X[num_cols] = X[num_cols].astype("float32")

        logging.info(
            f"Preprocessing complete. Shape X: {X.shape}, y: {y.shape}. "
            f"Categorical cols: {len(cat_cols)}, numeric cols: {len(X.columns) - len(cat_cols)}."
        )

        return X, y

    def run(self):
        """Train model and compute training scores + metadata."""
        self.X_train, self.y_train = self._preprocess_data(self.df_train)

        model = XGBRegressor(
            enable_categorical=True,
            **self.training_params
        )

        model.fit(self.X_train, self.y_train)

        y_pred = model.predict(self.X_train)

        scores = {
            "r2_score": float(r2_score(self.y_train, y_pred)),
            "rmse": float(root_mean_squared_error(self.y_train, y_pred))
        }

        metadata = {
            "model": "XGBRegressor",
            "params_used": self.training_params,
            "n_samples": len(self.X_train),
            "target_col": self.target_col,
            "enable_categorical": True
        }

        logging.info(f"Training complete. RMSE={scores['rmse']:.4f}, R²={scores['r2_score']:.4f}")

        return model, scores, metadata
