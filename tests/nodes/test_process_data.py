import pandas as pd
import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from edf_forecasting.pipelines.process_data.nodes import add_tempo_min
from edf_forecasting.components.eco2mix_add_features import Eco2mixFeaturesMinute


def test_add_tempo_min_calls_adder_correctly():
    params = {"mode": "aggregate_minute"}

    df_tempo = pd.DataFrame({
        "Datetime": pd.to_datetime([
            "2023-01-01 00:00",
            "2023-01-01 00:01",
            "2023-01-01 00:02",
            "2023-01-01 00:03",
            "2023-01-01 00:04",
        ]),
        "tempo": ["BLEU", "BLEU", "BLANC", "BLEU", "ROUGE"],
    })

    df_data = pd.DataFrame({
        "Datetime": pd.to_datetime([
            "2023-01-01 00:00",
            "2023-01-01 00:01",
            "2023-01-01 00:02",
            "2023-01-01 00:03",
            "2023-01-01 00:04",
        ]),
        "consumption": [100, 105, 98, 110, 108],
    })

    fake_result = df_data.assign(tempo="BLEU")

    with patch(
        "edf_forecasting.pipelines.process_data.nodes.Eco2MixAddTempo"
    ) as MockTempoAdder, \
        patch("edf_forecasting.pipelines.process_data.nodes.mlflow"):

        tempo_adder_instance = MagicMock()
        MockTempoAdder.return_value = tempo_adder_instance
        tempo_adder_instance.add_tempo.return_value = fake_result

        df = add_tempo_min(df_tempo, df_data, params)

        MockTempoAdder.assert_called_once()

def test_add_tempo_min_invalid_mode_raises():
    params = {"mode": "westeros"}

    df_tempo = pd.DataFrame({"Date": [1], "tempo": ["BLEU"]})
    df_data = pd.DataFrame({"Date": [1], "Consumption": [10]})

    with patch("edf_forecasting.pipelines.process_data.nodes.mlflow"):
        with pytest.raises(ValueError):
            add_tempo_min(df_tempo, df_data, params)

def test_run_adds_deterministic_time_features():
    df = pd.DataFrame({
        "Datetime": pd.to_datetime([
            "2023-01-02 10:00",  # lundi
            "2023-01-07 15:00",  # samedi
        ])
    })

    features = Eco2mixFeaturesMinute(df)
    result = features.run(include=["hour", "dayofweek", "is_weekend"])

    assert result.loc[0, "hour"] == 10
    assert result.loc[0, "dayofweek"] == 0
    assert result.loc[0, "is_weekend"] == 0

    assert result.loc[1, "hour"] == 15
    assert result.loc[1, "dayofweek"] == 5
    assert result.loc[1, "is_weekend"] == 1

def test_run_adds_vacation_flag_with_binary_values():
    df = pd.DataFrame({
        "Datetime": pd.to_datetime([
            "2023-01-01 12:00",  # jour férié
            "2023-01-03 12:00",
        ])
    })

    features = Eco2mixFeaturesMinute(df)
    result = features.run(include=["vacation"])

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
    mock_get.return_value.raise_for_status.return_value = None

    df = pd.DataFrame({
        "Datetime": pd.to_datetime(["2025-01-01T00:00"])
    })

    features = Eco2mixFeaturesMinute(df)
    result = features.run(include=["temperature"])

    mock_get.assert_called_once()
    assert "temperature_2m" in result.columns
