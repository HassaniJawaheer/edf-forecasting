from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from edf_forecasting_api.main import app
import pytest
pytestmark = pytest.mark.integration

def test_contrat_predict():
    client = TestClient(app)

    payload = {
        "features": [[75058.3]],
        "n_predictions": 1
    }

    fake_response = [[75059.3]]

    with patch("edf_forecasting_api.main.model_manager") as mock_model_manager, \
        patch("edf_forecasting_api.main.log_predictions"):
        
        mock_model_manager.predict.return_value = fake_response

        response = client.post("/predict", json=payload)
    
    assert response.status_code == 200
    body = response.json()
    assert "predictions" in body
    assert isinstance(body["predictions"], list)
    assert "prediction_id" in body

def test_route_feedback():
    client = TestClient(app)

    payload = {
        "prediction_id": "123abc",
        "inputs": [[78567.5]],
        "outputs": [[78596.2]]
    }
     
    with patch("edf_forecasting_api.main.log_feedback"):
        response = client.post("/feedback", json=payload)
    
    assert response.status_code == 200
    body = response.json()
    assert body["message"] == "Feedback saved"
