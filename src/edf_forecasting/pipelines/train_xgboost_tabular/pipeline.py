"""
This is a boilerplate pipeline 'train_xgboost_tabular'
generated using Kedro 1.0.0
"""

from kedro.pipeline import node, Pipeline, pipeline
from .nodes import train, calibrate, evaluate


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=train,
            inputs=["train_checked_consumption_data", "xgboost_tabular_optuna_30min_best_params", "params:train_tabular_30min"],
            outputs=["trained_model_tabular_xgboost_30min", "train_scores_tabular_xgboost_30min", "metadata_tabular_xgboost_30min"],
            name="train_tabular_30min"
        ),
        node(
            func=calibrate,
            inputs=["cal_checked_consumption_data", "trained_model_tabular_xgboost_30min", "params:calibration_tabular_30min"],
            outputs=["q_inf_tabular_xgboost_30min", "q_sup_tabular_xgboost_30min"],
            name="calibrate_tabular_30min"
        ),
        node(
            func=evaluate,
            inputs=[
                "trained_model_tabular_xgboost_30min",
                "test_checked_consumption_data",
                "q_inf_tabular_xgboost_30min",
                "q_sup_tabular_xgboost_30min",
                "params:evaluate_tabular_30min"
            ],
            outputs="test_scores_tabular_xgboost_30min",
            name="evaluate_tabular_30min"
        ),
    ])