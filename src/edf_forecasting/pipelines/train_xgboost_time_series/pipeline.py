"""
This is a boilerplate pipeline 'train_xgboost_time_series'
generated using Kedro 1.0.0
"""

from kedro.pipeline import node, Pipeline, pipeline
from .nodes import train, calibrate, evaluate


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=train,
            inputs=["train_checked_consumption_data", "xgboost_time_series_optuna_30min_best_params", "params:train"],
            outputs=["trained_model_time_series_xgboost_30min", "train_scores_time_series_xgboost_30min", "metadata_time_series_xgboost_30min"],
            name="train"
        ),
        node(
            func=calibrate,
            inputs=["cal_checked_consumption_data", "trained_model_time_series_xgboost_30min", "params:calibration"],
            outputs=["q_inf_time_series_xgboost_30min", "q_sup_time_series_xgboost_30min"],
            name="calibrate"
        ),
        node(
            func=evaluate,
            inputs=[
                "trained_model_time_series_xgboost_30min",
                "test_checked_consumption_data",
                "q_inf_time_series_xgboost_30min",
                "q_sup_time_series_xgboost_30min",
                "params:evaluate"
            ],
            outputs="test_scores_time_series_xgboost_30min",
            name="evaluate"
        ),
    ])