"""
This is a boilerplate pipeline 'create_reference_data'
generated using Kedro 1.0.0
"""
import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Any

def create_reference_data(df: pd.DataFrame, target_col: str, window_size: int = 48, fraction: Optional[int] = None) -> pd.DataFrame:
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in data")
    
    values = df[target_col].dropna().values
    X = np.lib.stride_tricks.sliding_window_view(values, window_shape=window_size)[:-1]
    y = values[window_size:]
    data = np.concatenate([X, y.reshape(-1, 1)], axis=1)

    if isinstance(fraction, int) and fraction < len(data):
        data = data[np.random.choice(len(data), size=fraction, replace=False)]
    
    columns = [f"consumption_{i + 1}" for i in range(window_size)] + ["target"]
    return pd.DataFrame(data, columns=columns)

def add_predictions_to_reference(df: pd.DataFrame, model: Any, window_size: int = 48):
    preds = []
    for _, row in df.iterrows():
        features = np.array([row[f"consumption_{i + 1}"] for i in range(window_size)]).reshape(1, -1)
        pred = model.predict(features)[0]
        preds.append(pred)
    
    df["prediction"] = preds
    return df

def update_latest_reference_symlink(reference_dir: str):
    latest_link = Path("./data/03_primary/eco2mix/latest_reference")
    if latest_link.exists() or latest_link.is_symlink():
        latest_link.unlink()
    os.symlink(reference_dir, latest_link)

def generate_reference_datasets(train_df: pd.DataFrame, test_df: pd.DataFrame, model: Any, params: dict) -> dict:
    reference_dir = Path(params["reference_dir"])
    reference_dir.mkdir(parents=True, exist_ok=True)
    
    ref_drift = create_reference_data(
        df=train_df,
        target_col=params["target_col"],
        window_size=params["window_size"],
        fraction=params.get("fraction"),
    )

    ref_perf = create_reference_data(
        df=test_df,
        target_col=params["target_col"],
        window_size=params["window_size"],
        fraction=params.get("fraction"),
    )

    ref_perf = add_predictions_to_reference(ref_perf, model=model, window_size=params["window_size"])

    update_latest_reference_symlink(str(reference_dir))

    return  ref_drift, ref_perf