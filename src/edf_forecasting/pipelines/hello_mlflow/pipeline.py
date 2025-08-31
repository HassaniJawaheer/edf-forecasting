"""
This is a boilerplate pipeline 'hello_mlflow'
generated using Kedro 1.0.0
"""

from kedro.pipeline import node, Pipeline  # noqa
from .nodes import hello


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        node(hello,
            "params:hello_message",
            "hello_output",
            name="hello"),
    ])
