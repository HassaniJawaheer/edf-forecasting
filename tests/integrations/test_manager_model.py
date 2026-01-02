import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from edf_forecasting_api.model_manager import ModelManager


def test_predict_autoregressive_shape():
    manager = ModelManager(model_name="timeseries_xgboost_30min")

    # Fake model
    fake_model = MagicMock()
    fake_model.predict.return_value = np.array([10.0])

    manager.model = fake_model

    X = [[1.0, 2.0]]
    n_predictions = 3

    preds = manager.predict(X, n_predictions)

    preds = np.array(preds)

    assert preds.shape == (1, 3)


def test_predict_without_model_raises():
    manager = ModelManager(model_name="timeseries_xgboost_30min")

    with pytest.raises(RuntimeError):
        manager.predict([[1.0, 2.0]], 1)

def test_load_model_no_production_version():
    manager = ModelManager(model_name="")

    with patch("mlflow.tracking.MlflowClient") as MockClient:
        client = MockClient.return_value
        client.search_model_versions.return_value = []

        manager.load_model()

    assert manager.model is None
    assert manager.current_version is None

def test_load_model_new_version_detected():
    manager = ModelManager(model_name="timeseries_xgboost_30min")

    fake_version = MagicMock()
    fake_version.version = "2"
    fake_version.current_stage = "Production"

    with patch("mlflow.tracking.MlflowClient") as MockClient, \
        patch("mlflow.pyfunc.load_model") as mock_load:

        client = MockClient.return_value
        client.search_model_versions.return_value = [fake_version]

        mock_model = MagicMock()
        mock_load.return_value = mock_model

        manager.load_model()
    
    assert manager.model is mock_model
    assert manager.current_version == "2"
