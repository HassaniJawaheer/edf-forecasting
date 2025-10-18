"""
This is a boilerplate pipeline 'tune_xgboost_time_series'
generated using Kedro 1.0.0
"""
import numpy as np
import mlflow
from edf_forecasting.components.eco2mix_tune_xgboost import XGBoostTuner


def create_windows(df_data, params):
    """Create sliding windows of 48 half-hours from the training dataset."""
    df = df_data.copy()
    window_size = params["window_size"]
    target_col = params["target_col"]

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found.")

    values = df[target_col].values
    if len(values) <= window_size:
        raise ValueError("Insufficient data to create at least one window.")

    X = np.lib.stride_tricks.sliding_window_view(values, window_shape=window_size)[:-1]
    y = values[window_size:]

    mlflow.log_params({
        "window_size": window_size,
        "target_col": target_col,
        "n_samples": len(y)
    })

    return X, y

def tune(X, y, params):
    """Run Optuna hyperparameter tuning for XGBoost."""
    tuner = XGBoostTuner(
        n_trials=params["n_trials"],
        timeout=params["timeout"],
        cv=params["cv"],
        seed=params["seed"]
    )

    best_params, study = tuner.run(X, y)

    mlflow.log_params(best_params)
    mlflow.log_metric("n_trials_done", len(study.trials))

    return best_params
