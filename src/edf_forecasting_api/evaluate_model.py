import pandas as pd
import numpy as np
import requests
import logging
from pathlib import Path

API_URL_PREDICT = "http://127.0.0.1:8000/predict"
API_URL_FEEDBACK = "http://127.0.0.1:8000/feedback"
WINDOW_SIZE = 48

def read_df(path: Path) -> pd.DataFrame:
    """Read a data file as CSV with tab separator and latin1 encoding."""
    return pd.read_csv(path, sep="\t", encoding="latin1", index_col=False, low_memory=False)

def create_sliding_windows(values, window_size=48):
    X = np.lib.stride_tricks.sliding_window_view(values, window_shape=window_size)[:-1]
    y = values[window_size:]
    return X, y

def simulate_predictions_and_feedbacks(df: pd.DataFrame, target_col="Consommation", limit=None):
    values = df[target_col].values
    X, y = create_sliding_windows(values, WINDOW_SIZE)
    if limit:
        X, y = X[:limit], y[:limit]

    all_preds = []
    all_feedbacks = []

    for i, (features, true_val) in enumerate(zip(X, y)):
        payload_pred = {"features": [features.tolist()], "n_predictions": 1}
        try:
            response = requests.post(API_URL_PREDICT, json=payload_pred)
            response.raise_for_status()
            result = response.json()
            preds = result.get("predictions", [[None]])[0]
            prediction_id = result.get("prediction_id")

            all_preds.append({
                "prediction_id": prediction_id,
                "features": features.tolist(),
                "prediction": preds[0],
                "target": true_val
            })

            feedback_payload = {
                "prediction_id": prediction_id,
                "inputs": [features.tolist()],
                "outputs": [[true_val]]
            }

            fb_response = requests.post(API_URL_FEEDBACK, json=feedback_payload)
            fb_response.raise_for_status()

            all_feedbacks.append(feedback_payload)
            logging.info(f"Sent feedback {i+1}/{len(X)} | ID: {prediction_id}")
        except Exception as e:
            logging.error(f"Error at index {i}: {e}")
            continue
    df_results = pd.DataFrame(all_preds)
    return df_results, all_feedbacks

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    data = read_df(path="./src/edf_forecasting_api/data/eCO2mix_RTE_Annuel-Definitif_2023.xls")
    data = data.dropna(subset=["Consommation"]).reset_index(drop=True)
  
    results_df, feedbacks = simulate_predictions_and_feedbacks(data, target_col="Consommation", limit=None)

    print("\n Simulation finished.")
    print(f"Predictions generated: {len(results_df)}")
    print(f"Feedbacks sent: {len(feedbacks)}")