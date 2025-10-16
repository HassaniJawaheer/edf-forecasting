"""
This is a boilerplate pipeline 'train_xgboost_30min'
generated using Kedro 1.0.0
"""

from kedro.pipeline import node, Pipeline, pipeline
from .nodes import train, calibrate, evaluate


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=train,
            inputs=["train_checked_consumption_data", "xgboost_optuna_best_params_30min", "params:train"],
            outputs=["trained_model_xgboost_30min", "train_scores_xgboost_30min", "metadata_xgboost_30min"],
            name="train"
        ),
        node(
            func=calibrate,
            inputs=["cal_checked_consumption_data", "trained_model_xgboost_30min", "params:calibration"],
            outputs=["q_inf_xgboost_30min", "q_sup_xgboost_30min"],
            name="calibrate"
        ),
        node(
            func=evaluate,
            inputs=[
                "trained_model_xgboost_30min",
                "test_checked_consumption_data",
                "q_inf_xgboost_30min",
                "q_sup_xgboost_30min",
                "params:evaluate"
            ],
            outputs="test_scores_xgboost_30min",
            name="evaluate"
        ),
    ])