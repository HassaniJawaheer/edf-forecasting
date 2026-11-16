from kedro.pipeline import Pipeline, node, pipeline

from .nodes import run_backtest


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                run_backtest,
                [
                    "backtest_model",
                    "backtest_history",
                    "backtest_evaluation",
                    "params:backtest",
                ],
                None,
                name="evaluate_recursive_forecasts",
            )
        ]
    )
