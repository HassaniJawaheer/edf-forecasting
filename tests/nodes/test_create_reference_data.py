from edf_forecasting.pipelines.create_reference_data.nodes import (
    create_reference_data,
    add_predictions_to_reference
)
import pandas as pd
import numpy as np
import pytest
from unittest.mock import MagicMock


def test_create_reference_data_missing_target():
    df = pd.DataFrame({"a": range(10)})
    with pytest.raises(ValueError):
        create_reference_data(df, target_col="consoumption")

def test_create_reference_data_shape_and_columns():
    df = pd.DataFrame({"consumption": np.arange(60)})
    window_size = 10

    ref = create_reference_data(
        df=df,
        target_col="consumption",
        window_size=window_size
    )

    assert ref.shape[1] == window_size + 1
    assert ref.columns[-1] == "target"
    assert len(ref) == 60 - window_size

def test_create_reference_data_target_alignement():
    df = pd.DataFrame({"consumption": np.arange(20)})
    window_size = 5

    ref = create_reference_data(
        df,
        target_col="consumption",
        window_size=window_size
    )

    first_row = ref.iloc[0]
    assert first_row["target"] == df["consumption"].iloc[window_size]


def test_create_reference_data_fraction():
    df = pd.DataFrame({"consumption": np.arange(100)})

    ref = create_reference_data(
        df=df,
        target_col="consumption",
        window_size=10,
        fraction=5,
    )

    assert len(ref) == 5


def test_add_predictions_to_reference():
    df = pd.DataFrame({
        f"consumption_{i+1}": [i] for i in range(5)
    })
    df["target"] = 1.0

    model = MagicMock()
    model.predict.return_value = np.array([42.0])

    out = add_predictions_to_reference(df, model, window_size=5)

    assert "prediction" in out.columns
    assert out["prediction"].iloc[0] == 42.0
    model.predict.assert_called()