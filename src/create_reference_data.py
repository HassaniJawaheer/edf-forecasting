import os
import pandas as pd
import numpy as np
import requests
from typing import Optional, Any

def create_reference_data(
    df: pd.DataFrame,
    target_col: str,
    window_size: int,
    fraction: Optional[int] = None
) -> pd.DataFrame:
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in data")

    values = df[target_col].values
    X = np.lib.stride_tricks.sliding_window_view(values, window_shape=window_size)[:-1]
    y = values[window_size:]
    data = np.concatenate([X, y.reshape(-1, 1)], axis=1)

    if isinstance(fraction, int) and fraction < len(data):
        data = data[np.random.choice(len(data), size=fraction, replace=False)]

    columns = [f"consumption_{i+1}" for i in range(window_size)] + ["target"]
    return pd.DataFrame(data, columns=columns)


def add_predictions_to_reference(
    df: pd.DataFrame,
    model: Optional[Any] = None,
    api_url: Optional[str] = None,
    window_size: int = 48
) -> pd.DataFrame:
    preds = []
    for _, row in df.iterrows():
        features = np.array([row[f"consumption_{i+1}"] for i in range(window_size)]).reshape(1, -1)

        if model is not None:
            pred = model.predict(features)[0]
        elif api_url is not None:
            payload = {"features": features.tolist(), "n_predictions": 1}
            response = requests.post(api_url, json=payload)
            response.raise_for_status()
            result = response.json()
            pred = result.get("predictions", [[None]])[0][0]
        else:
            raise ValueError("Either `model` or `api_url` must be provided.")

        preds.append(pred)

    df["prediction"] = preds
    return df


if __name__ == "__main__":
    train_path = "./data/03_primary/eco2mix/definitive/30min/checked/split/train_checked_consumption_data.csv/2025-10-24T08.12.07.064Z/train_checked_consumption_data.csv"
    test_path = "./data/03_primary/eco2mix/definitive/30min/checked/split/test_checked_consumption_data.csv/2025-10-24T08.12.07.064Z/test_checked_consumption_data.csv"
    reference_dir = "./data/03_primary/eco2mix/definitive/30min/checked/reference"
    os.makedirs(reference_dir, exist_ok=True)

    train_df = pd.read_csv(train_path, low_memory=False)
    test_df = pd.read_csv(test_path, low_memory=False)

    ref_drift = create_reference_data(train_df, target_col="Consommation", window_size=48, fraction=10000)
    ref_perf = create_reference_data(test_df, target_col="Consommation", window_size=48, fraction=10000)

    ref_perf = add_predictions_to_reference(ref_perf, api_url="http://127.0.0.1:8000/predict")

    ref_drift.to_csv(os.path.join(reference_dir, "reference_data_drift.csv"), index=False)
    ref_perf.to_csv(os.path.join(reference_dir, "reference_data_perf.csv"), index=False)
