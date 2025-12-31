"""
This is a boilerplate test file for pipeline 'create_reference_data'
generated using Kedro 1.0.0.
Please add your pipeline tests here.

Kedro recommends using `pytest` framework, more info about it can be found
in the official documentation:
https://docs.pytest.org/en/latest/getting-started.html
"""

from edf_forecasting.pipelines.create_reference_data.pipeline import create_pipeline

def test_pipeline_creation():
    pipeline = create_pipeline()
    assert pipeline is not None
    assert len(pipeline.nodes) >0