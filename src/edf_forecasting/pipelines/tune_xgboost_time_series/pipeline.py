"""
This is a boilerplate pipeline 'tune_xgboost_time_series'
generated using Kedro 1.0.0
"""

from kedro.pipeline import node, Pipeline, pipeline  # noqa
from .nodes import *

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=create_windows,
            inputs=["train_checked_consumption_data", "params:create_windows"],
            outputs=["X_train", "y_train"],
            name="create_windows"
        ),
        node(
            func=tune,
            inputs=["X_train", "y_train", "params:tune"],
            outputs="xgboost_time_series_optuna_30min_best_params",
            name="tune"
        )
    ])