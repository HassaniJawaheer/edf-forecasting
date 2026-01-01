import pandas as pd
import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from edf_forecasting.pipelines.process_data.nodes import (
    add_tempo_min
)
from edf_forecasting.components.eco2mix_add_features import Eco2mixFeaturesMinute


def test_add_tempo_valid_dataframe():
    
    params = {
        "mode": "aggregate_minute"
    }
    
    df_tempo = pd.DataFrame({
        "Date" : np.arange(5),
        "tempo": np.random.choice(["BLEU", "BLANC", "ROUGE"], size=5)
    })
    df_data = pd.DataFrame({
        "Date" : np.arange(5),
        "Comsumption": np.arange(5)
    })
    
    # Mocking Eco2MixAddTempo
    with patch("edf_forecasting.components.eco2mix_add_tempo.Eco2MixAddTempo") as MockTempoAdder, \
        patch("edf_forecasting.pipelines.process_data.nodes.mlflow"):
        tempo_adder_instance = MagicMock()
        MockTempoAdder.return_value = tempo_adder_instance
        
        # Vérifie si le merge se passe bien
        df = add_tempo_min(df_tempo, df_data, params)
        assert isinstance(df, pd.DataFrame)

        # Vérifie si tempo est bien présent sur le résultats
        assert "tempo" in df.columns
        assert df.shape[0] == df_data.shape[0]


def test_add_tempo_valid_mode():

    params = {
        "mode": "westeros"
    }

    df_tempo = pd.DataFrame({
        "Date" : np.arange(5),
        "tempo": np.random.choice(["BLEU", "BLANC", "ROUGE"], size=5)
    })
    df_data = pd.DataFrame({
        "Date" : np.arange(5),
        "Comsumption": np.arange(5)
    })

    # Mocking Eco2MixAddTempo
    with patch("edf_forecasting.components.eco2mix_add_tempo.Eco2MixAddTempo") as MockTempoAdder, \
        patch("edf_forecasting.pipelines.process_data.nodes.mlflow"):
        tempo_adder_instance = MagicMock()
        MockTempoAdder.return_value = tempo_adder_instance
        
        # Vérifie que mode invalide ne passe pas
        with pytest.raises(ValueError):
            df = add_tempo_min(df_tempo, df_data, params)


def test_run_adds_deterministic_time_features():
    df = pd.DataFrame({
        "Datetime": pd.to_datetime([
            "2023-01-02 10:00",  # lundi
            "2023-01-07 15:00",  # samedi
        ])
    })

    features = Eco2mixFeaturesMinute(df)
    result = features.run(include=["hour", "dayofweek", "is_weekend"])

    assert "hour" in result.columns
    assert "dayofweek" in result.columns
    assert "is_weekend" in result.columns

    assert result.loc[0, "hour"] == 10
    assert result.loc[0, "dayofweek"] == 0
    assert result.loc[0, "is_weekend"] == 0

    assert result.loc[1, "hour"] == 15
    assert result.loc[1, "dayofweek"] == 5
    assert result.loc[1, "is_weekend"] == 1


def test_run_adds_vacation_flag_with_binary_values():
    import pandas as pd
    from edf_forecasting.components.eco2mix_add_features import Eco2mixFeaturesMinute

    df = pd.DataFrame({
        "Datetime": pd.to_datetime([
            "2023-01-01 12:00",  # jour férié (Nouvel An)
            "2023-01-03 12:00",  # jour normal
        ])
    })

    features = Eco2mixFeaturesMinute(df)
    result = features.run(include=["vacation"])

    assert "is_vacation" in result.columns
    # Est ce que qoutes les valeurs observées appartienne à l'ensemble {0,1}.
    assert set(result["is_vacation"].unique()).issubset({0, 1})

    assert result.loc[0, "is_vacation"] == 1
    assert result.loc[1, "is_vacation"] == 0


def test_run_temperature_feature_calls_api_and_adds_column(mocker):
    
    mock_response = {
        "hourly": {
            "time": ["2025-01-01T00:00"],
            "temperature_2m": [5.0],
        }
    }

    mock_get = mocker.patch("requests.get")
    mock_get.return_value.json.return_value = mock_response
    mock_get.return_value.rise_for_status.return_value = None

    df = pd.DataFrame({
        "Datetime": pd.to_datetime(["2025-01-01T00:00"])
    })


    features = Eco2mixFeaturesMinute(df)
    result = features.run(include=["temperature"])

    mock_get.assert_called_once()
    assert "temperature_2m" in result.columns
