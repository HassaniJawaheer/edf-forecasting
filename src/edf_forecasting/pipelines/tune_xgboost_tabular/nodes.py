"""
This is a boilerplate pipeline 'tune_xgboost_tabular'
generated using Kedro 1.0.0
"""
import mlflow
from edf_forecasting.components.eco2mix_preprocess_xgboost_30min import Eco2mixPreprocessXGBoost30min
from edf_forecasting.components.eco2mix_tune_xgboost import XGBoostTuner

def preprocess_data(df, params):
    """ Preprocess tabular data for Xgboost training."""
    df = df.copy()
    processor = Eco2mixPreprocessXGBoost30min(
        target_col=params["target_col"],
        drop_duplicates=params["drop_duplicates"],
        ensure_float32=params["ensure_float32"]
    )

    X, y = processor.run(df)
    return X, y

def tune_xgboost_tabular(X, y, params):
    "Run Optuna hyperparameter tuning for XGBoost."
    tuner = XGBoostTuner(
        dir=params["dir"],
        n_trials=params["n_trials"],
        timeout=params["timeout"],
        cv=params["cv"],
        seed=params["seed"]
    )

    best_params, study = tuner.run(X,y)

    mlflow.log_params(best_params)
    mlflow.log_metric("n_trials_done", len(study.trials))

    return best_params