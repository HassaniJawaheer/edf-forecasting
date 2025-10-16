"""
This is a boilerplate pipeline 'train_xgboost_30min'
generated using Kedro 1.0.0
"""
import mlflow
import json
import tempfile
import os
from edf_forecasting.components.eco2mix_evaluate_xgboost_30min import XGBEvaluate30min
from edf_forecasting.components.eco2mix_calibrate_xgboost_30min import XGBCalibrator30min
from edf_forecasting.components.eco2mix_train_xgboost_30min import Eco2mixTrainGBoost30min


def train(df_train, training_params, params):
    """Train the XGBoost model on 30-minute consumption data."""
    mlflow.log_params({
        "train.window_size": params["windows_size"],
        "train.target_col": params["target_col"]
    })

    trainer = Eco2mixTrainGBoost30min(
        df_train=df_train,
        training_params=training_params,
        windows_size=params["windows_size"],
        target_col=params["target_col"]
    )

    model, scores, metadata = trainer.run()

    mlflow.xgboost.log_model(
        xgb_model=model,
        artifact_path="model",
        registered_model_name="timeseries_xgboost_30min"
    )

    mlflow.log_metrics({f"train.{k}": v for k, v in scores.items() if isinstance(v, (int, float))})
    return model, scores, metadata


def calibrate(df_data, model, params):
    """Calibrate model prediction intervals using calibration dataset."""
    calibrator = XGBCalibrator30min(
        df_cal=df_data,
        model=model,
        error_type=params["error_type"],
        windows_size=params["windows_size"],
        target_col=params["target_col"]
    )

    q_inf, q_sup = calibrator.run(alpha=params["alpha"])

    mlflow.log_param("calibration.alpha", params["alpha"])
    mlflow.log_metric("calibration.q_inf_mean", float(q_inf))
    mlflow.log_metric("calibration.q_sup_mean", float(q_sup))
    return q_inf, q_sup


def evaluate(model, df_test, q_inf, q_sup, params):
    """Evaluate model on test dataset using calibrated quantiles."""
    evaluator = XGBEvaluate30min(
        model=model,
        df_test=df_test,
        q_inf=q_inf,
        q_sup=q_sup,
        quantile=params["quantile"],
        windows_size=params["windows_size"],
        target_col=params["target_col"]
    )

    results = evaluator.run()
    scalars = {k: v for k, v in results.items() if isinstance(v, (int, float))}
    if scalars:
        mlflow.log_metrics({f"eval.{k}": v for k, v in scalars.items()})

    # save full results as MLflow artifact
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "evaluation_results.json")
        with open(path, "w") as f:
            json.dump(results, f, indent=2, default=float)
        mlflow.log_artifact(path, artifact_path="evaluation")

    return results