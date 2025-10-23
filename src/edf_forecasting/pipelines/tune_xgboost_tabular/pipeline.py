"""
This is a boilerplate pipeline 'tune_xgboost_tabular'
generated using Kedro 1.0.0
"""

from kedro.pipeline import node, Pipeline  # noqa
from .nodes import *


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        node(
            func=preprocess_data,
            inputs=["train_checked_consumption_data", "params:preprocess"],
            outputs=["X_train_xgboost_tabular", "y_train_xgboost_tabular"],
            name="preprocess"
        ),
        node(
            func=tune_xgboost_tabular,
            inputs=["X_train_xgboost_tabular", "y_train_xgboost_tabular", "params:tune_xgboost_tabular"],
            outputs="xgboost_tabular_optuna_30min_best_params",
            name="tune_xgboost_tabular"
        )
    ])
