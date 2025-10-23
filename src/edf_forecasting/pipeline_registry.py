"""Project pipelines."""
from __future__ import annotations

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline


def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    pipelines = find_pipelines()
    
    #pipelines["__default__"] = sum(pipelines.values())
    pipelines["full"] = (
        pipelines["fetch_raw_data"]
        + pipelines["process_data"]
        + pipelines["tune_xgboost_time_series"]
        + pipelines["train_xgboost_time_series"]
    )

    pipelines["__default__"] = pipelines["full"]

    return pipelines
