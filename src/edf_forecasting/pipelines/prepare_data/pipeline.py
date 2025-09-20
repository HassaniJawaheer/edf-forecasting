"""
This is a boilerplate pipeline 'prepare_data'
generated using Kedro 1.0.0
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import scrape_data, prestructure_data, clean_data

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=scrape_data,
            inputs="params:scraping",
            outputs="scraping_status",
            name="scrape_data",
        ),
        node(
            func=prestructure_data,
            inputs=["scraping_status", "params:prestructuring"],
            outputs="prestructuring_status",
            name="prestructure_data",
        ),
        node(
            func=clean_data,
            inputs=[
                "prestructuring_status",
                "consumption_data",
                "tempo_calendar",
                "params:cleaning",
            ],
            outputs=["cleaned_consumption_data", "cleaned_tempo_calendar"],
            name="clean_data",
        ),
    ])
