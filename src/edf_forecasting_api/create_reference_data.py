import pandas as pd
import numpy as np
import os

train_data_dir = "./data/03_primary/eco2mix/definitive/30min/checked/split/train_checked_consumption_data.csv/2025-10-24T08.12.07.064Z/train_checked_consumption_data.csv"
reference_data_dir = "./data/03_primary/eco2mix/definitive/30min/checked/reference/data.csv"
fraction = 10000
target_col = "Consommation"
window_size = 48
# Create reference repo
os.makedirs("./data/03_primary/eco2mix/definitive/30min/checked/reference", exist_ok=True)


def create_data():
    df = pd.read_csv(train_data_dir)
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in data")
    values = df[target_col].values
    X = np.lib.stride_tricks.sliding_window_view(values, window_shape=window_size)[:-1]
    y = values[window_size:]
    data = np.concatenate([X, y.reshape(-1, 1)], axis=1)
    if isinstance(fraction, int) and fraction < len(data):
        data = data[np.random.choice(len(data), size=fraction, replace=False)]
    return data

if __name__ == "__main__":
    data = create_data()
    pd.DataFrame(data).to_csv(reference_data_dir, index=False)
    print(f"Reference data saved to {reference_data_dir} (shape: {data.shape})")

