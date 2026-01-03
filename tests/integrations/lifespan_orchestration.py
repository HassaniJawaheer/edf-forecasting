from fastapi.testclient import TestClient
from unittest.mock import patch
from edf_forecasting_api.main import app


def test_api_startup_calls():

    with patch("edf_forecasting_api.main.model_manager.load_model") as load_model, \
         patch("edf_forecasting_api.main.model_manager.start_watcher") as start_watcher, \
         patch("edf_forecasting_api.main.schedule_monitoring") as schedule_monitoring, \
         patch("edf_forecasting_api.main.MetricsStorage"):

        with TestClient(app):
            pass  # lifespan startup + shutdown

        load_model.assert_called_once()
        start_watcher.assert_called_once()
        schedule_monitoring.assert_called_once()


def test_api_shutdown_calls():

    with patch("edf_forecasting_api.main.model_manager.load_model"), \
         patch("edf_forecasting_api.main.model_manager.start_watcher"), \
         patch("edf_forecasting_api.main.schedule_monitoring"), \
         patch("edf_forecasting_api.main.MetricsStorage"), \
         patch("edf_forecasting_api.main.model_manager.stop_watcher") as stop_watcher:

        with TestClient(app):
            pass 

        stop_watcher.assert_called_once()







