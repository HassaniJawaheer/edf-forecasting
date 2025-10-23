"""
This is a boilerplate pipeline 'train_xgboost_tabular'
generated using Kedro 1.0.0
"""
import mlflow
import json
import os
import numpy as np
import subprocess
from edf_forecasting.components.eco2mix_calibrate_xgboost_tabular_30min import Eco2mixCalibrateTabXGBoost30min
from edf_forecasting.components.eco2mix_evaluate_xgboost_tabular_30min import Eco2mixEvaluateTabXGBoost30min
from edf_forecasting.components.eco2mix_train_xgboost_tabular_30min import Eco2mixTrainTabXGBoost30min


def train(df_train, training_params, params):
    
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    mlflow.set_tag("git_commit", commit)

    mlflow.log_params({
        "train.target_col": params["target_col"],
        "drop_duplicates": params["drop_duplicates"],
        "ensure_float32": params["ensure_float32"]
    })

    trainer = Eco2mixTrainTabXGBoost30min(
        df_train=df_train,
        training_params=training_params,
        target_col=params["target_col"],
        drop_duplicates=params["drop_duplicates"],
        ensure_float32=params["ensure_float32"]
    )

    model, scores, metadata = trainer.run()

    for k, v in scores.items():
        if isinstance(v, (int, float, np.floating)):
            mlflow.log_metric(f"train.{k}", float(v))

    mlflow.xgboost.log_model(
        xgb_model=model,
        artifact_path="model",
        registered_model_name="tabular_xgboost_30min"
    )

    mlflow.log_dict(metadata, "model_metadata.json")

    return model, scores, metadata


def calibrate(df_data, model, params):
    calibrator = Eco2mixCalibrateTabXGBoost30min(
        df_cal=df_data,
        model=model,
        error_type=params["error_type"],
        target_col=params["target_col"],
        drop_duplicates=params["drop_duplicates"],
        ensure_float32=params["ensure_float32"]
    )

    q_inf, q_sup = calibrator.run(alpha=params["alpha"])

    mlflow.log_params({
        "calibration.alpha": params["alpha"],
        "calibration.error_type": params["error_type"]
    })
    mlflow.log_metrics({
        "calibration.q_inf_mean": float(q_inf),
        "calibration.q_sup_mean": float(q_sup)
    })

    return q_inf, q_sup


def evaluate(model, df_test, q_inf, q_sup, params):
    evaluator = Eco2mixEvaluateTabXGBoost30min(
        model=model,
        df_test=df_test,
        q_inf=q_inf,
        q_sup=q_sup,
        quantile=params["quantile"],
        target_col=params["target_col"],
        drop_duplicates=params["drop_duplicates"],
        ensure_float32=params["ensure_float32"]
    )

    results = evaluator.run()

    for k, v in results.items():
        if isinstance(v, (int, float, np.floating)):
            mlflow.log_metric(f"eval.{k}", float(v))

    save_dir = "data/07_model_output/eco2mix/tabular/30min/xgboost/evaluation"
    os.makedirs(save_dir, exist_ok=True)
    eval_path = os.path.join(save_dir, "evaluation_results.json")

    with open(eval_path, "w") as f:
        json.dump(results, f, indent=2)

    mlflow.log_artifact(eval_path, artifact_path="evaluation")

    return results
