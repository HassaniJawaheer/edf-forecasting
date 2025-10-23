import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)

class Eco2mixPreprocessXGBoost30min:
    def __init__(
        self,
        target_col: str = "Consommation",
        drop_duplicates: bool = True,
        ensure_float32: bool = True,
    ):
        self.target_col = target_col
        self.drop_duplicates = drop_duplicates
        self.ensure_float32 = ensure_float32

    def run(self, df: pd.DataFrame):
        df = df.copy()
        logging.info("Starting preprocessing for XGBoost (30min)...")

        if self.drop_duplicates:
            df = df.drop_duplicates().reset_index(drop=True)

        if self.target_col not in df.columns:
            raise ValueError(f"Target column '{self.target_col}' not found in dataframe.")

        # Separate features and target
        y = df[self.target_col]
        X = df.drop(columns=[self.target_col])

        # Convert object columns to category for XGBoost native categorical handling
        cat_cols = X.select_dtypes(include=["object"]).columns
        for col in cat_cols:
            X[col] = X[col].astype("category")

        # Convert numerical columns to float32 for better XGBoost performance
        if self.ensure_float32:
            num_cols = X.select_dtypes(include=["int", "float"]).columns
            X[num_cols] = X[num_cols].astype("float32")

        logging.info(
            f"Preprocessing complete. Shape X: {X.shape}, y: {y.shape}. "
            f"Categorical cols: {len(cat_cols)}, numeric cols: {len(X.columns) - len(cat_cols)}."
        )

        return pd.DataFrame(X), pd.Series(y)
