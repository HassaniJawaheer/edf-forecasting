"""
This is a boilerplate pipeline 'process_data'
generated using Kedro 1.0.0
"""

from kedro.pipeline import Pipeline, node, pipeline # noqa
from .nodes import add_tempo_min, add_features_min, check_frequency, split_train_cal_test


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=add_tempo_min,
            inputs=["cleaned_consumption_data", "cleaned_tempo_calendar", "params:add_tempo"],
            outputs="tempo_consumption_data",
            name="add_tempo_min",
        ),
        node(
            func=add_features_min,
            inputs=["tempo_consumption_data", "params:add_features"],
            outputs="tempo_consumption_enriched_data",
            name="add_features_min",
        ),
        node(
            func=check_frequency,
            inputs=["tempo_consumption_enriched_data", "params:check_frequency"],
            outputs="checked_consumption_data",
            name="check_frequency",
        ),
        node(
            func=split_train_cal_test,
            inputs=["checked_consumption_data", "params:split"],
            outputs=["train_checked_consumption_data", "cal_checked_consumption_data", "test_checked_consumption_data"],
            name="split_train_cal_test",
        ),
    ])
