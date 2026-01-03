import pandas as pd
from edf_forecasting_api.monitoring.ml_monitoring import flatten_predictions
from edf_forecasting_api.monitoring.metrics_storage import MetricsStorage


def test_flatten_predictions():
    df = pd.DataFrame({
        "prediction_id": ["abc123"],
        "inputs": [[[78945.1, 61237.5]]],
        "outputs": [[[12345.6]]],
    })

    df_result = flatten_predictions(df)

    assert isinstance(df_result, pd.DataFrame)
    assert not df_result.empty
    assert "prediction_id" in df_result.columns
    assert "target" in df_result.columns

def test_metrics_storage_init_and_store(tmp_path):
    db_path = tmp_path / "metrics.db"
    storage = MetricsStorage(db_path=str(db_path))

    metrics = {
        "timestamp": "2025-01-01",
        "model_name": "test_model",
        "model_version": "1",
        "mae": 1.0,
        "rmse": 1.2,
        "r2": 0.9,
        "drift_score": 0.1,
        "drift_report_path": "x.html",
        "perf_report_path": "y.html",
    }

    # anti-crash, anti-régression
    storage.store_metrics(metrics)
