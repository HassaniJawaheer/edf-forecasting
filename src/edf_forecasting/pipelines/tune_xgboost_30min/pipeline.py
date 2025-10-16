"""
This is a boilerplate pipeline 'tune_xgboost_30min'
generated using Kedro 1.0.0
"""

from kedro.pipeline import node, Pipeline, pipeline  # noqa
from .nodes import *

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=create_windows,
            inputs=["train_checked_consumption_data", "params:create_windows"],
            outputs=["X_train_30min", "y_train_30min"],
            name="create_windows"
        ),
        node(
            func=tune,
            inputs=["X_train_30min", "y_train_30min", "params:tune"],
            outputs="xgboost_optuna_best_params_30min",
            name="tune"
        )
    ])