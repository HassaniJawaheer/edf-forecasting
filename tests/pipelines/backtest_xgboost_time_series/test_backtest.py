import numpy as np
import pandas as pd
import pytest

from edf_forecasting.pipelines.backtest_xgboost_time_series.nodes import (
    forecast,
    metrics,
    prepare,
)


class IncrementModel:
    def predict(self, x):
        return x[:, -1] + 1


def test_recursive_forecast_does_not_use_future_observations():
    values = np.arange(800, dtype=float)
    values[400:] = -500
    result = forecast(IncrementModel(), values, np.array([400]), 48, 60)
    np.testing.assert_array_equal(result["recursive"][0], np.arange(400, 460))
    assert result["updated_observations"][0, 0] == 400
    assert result["updated_observations"][0, 1] == -499
    np.testing.assert_array_equal(result["previous_day"][0, :48], values[352:400])
    np.testing.assert_array_equal(result["previous_day"][0, 48:], values[352:364])
    np.testing.assert_array_equal(result["previous_week"][0], values[64:124])
    np.testing.assert_array_equal(result["persistence"], np.full((1, 60), 399))


def test_weekly_baseline_repeats_only_available_history():
    values = np.arange(1200, dtype=float)
    result = forecast(IncrementModel(), values, np.array([400]), 48, 340)
    np.testing.assert_array_equal(result["previous_week"][0, 336:], values[64:68])


def test_metrics_use_signed_bias_and_root_mean_squared_error():
    frame = pd.DataFrame(
        {
            "method": ["a", "a"],
            "error": [-2, 4],
            "absolute_error": [2, 4],
            "squared_error": [4, 16],
        }
    )
    result = metrics(frame, ["method"]).iloc[0]
    assert result.mae_mw == 3
    assert result.bias_mw == 1
    assert result.rmse_mw == pytest.approx(np.sqrt(10))


def test_prepare_uses_previous_year_history_and_rejects_gaps():
    index = pd.date_range("2020-12-24", "2021-01-02 23:30", freq="30min")
    frame = pd.DataFrame({"Consommation": np.arange(len(index))}, index=index)
    history, evaluation = frame.loc["2020"], frame.loc["2021"]
    params = {"target": "Consommation", "year": 2021, "origin_hour": 0, "window": 48}
    timestamps, _, origins = prepare(history, evaluation, params)
    assert list(timestamps[origins]) == [
        pd.Timestamp("2021-01-01"),
        pd.Timestamp("2021-01-02"),
    ]
    with pytest.raises(ValueError, match="continuous"):
        prepare(history, evaluation.drop(evaluation.index[4]), params)
