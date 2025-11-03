"""
This is a boilerplate pipeline 'create_reference_data'
generated using Kedro 1.0.0
"""

from kedro.pipeline import node, Pipeline  # noqa
from .nodes import generate_reference_datasets

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=generate_reference_datasets,
                inputs=["train_checked_consumption_data", "test_checked_consumption_data", "trained_model_time_series_xgboost_30min", "params:create_reference_data"],
                outputs=["reference_data_drift", "reference_data_perf"],
                name="generate_reference_datasets",
            ),
        ]
    )
