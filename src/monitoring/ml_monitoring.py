import os
import json
import glob
import logging
from datetime import datetime

import pandas as pd
from evidently import Dataset, Report, DataDefinition, Regression
from evidently.presets import DataDriftPreset, RegressionPreset

from src.monitoring.metrics_storage import MetricsStorage


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

LOG_DIR = os.getenv("LOG_DIR", "src/logs")
REFERENCE_DRIFT = os.getenv("REFERENCE_DRIFT", "./data/03_primary/eco2mix/definitive/30min/checked/reference/reference_data_drift.csv")
REFERENCE_PERF = os.getenv("REFERENCE_PERF", "./data/03_primary/eco2mix/definitive/30min/checked/reference/reference_data_perf.csv")

PREDICTION_LOG = os.path.join(LOG_DIR, "predictions.jsonl")
FEEDBACK_LOG = os.path.join(LOG_DIR, "ground_truth.jsonl")

DRIFT_CRITICAL_THRESHOLD = float(os.getenv("DRIFT_CRITICAL_THRESHOLD", "0.5"))
RMSE_WARNING_THRESHOLD = float(os.getenv("RMSE_WARNING_THRESHOLD", "500"))


def get_latest_versioned_file(base_path: str) -> str:
    version_dirs = glob.glob(os.path.join(base_path, "*"))

    if not version_dirs:
        raise FileNotFoundError(f"No versioned files found for {base_path}")

    latest_version_dir = sorted(version_dirs)[-1]

    target_file = os.path.join(latest_version_dir, os.path.basename(base_path))

    if not os.path.isfile(target_file):
        raise FileNotFoundError(f"No csv file found at {target_file}")

    return target_file


def flatten_predictions(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for _, row in df.iterrows():
        inputs = row.get("inputs", [])
        outputs = row.get("outputs", [])

        if not isinstance(inputs, list) or not isinstance(outputs, list):
            continue

        for idx, (inp, out) in enumerate(zip(inputs, outputs)):
            if not isinstance(inp, list):
                continue

            first_val = out[0] if isinstance(out, list) and len(out) > 0 else out

            entry = {f"consumption_{i + 1}": v for i, v in enumerate(inp)}
            entry.update(
                {
                    "prediction_id": f"{row['prediction_id']}_{idx}",
                    "target": first_val,
                }
            )
            rows.append(entry)

    flat_df = pd.DataFrame(rows)
    return flat_df

def compute_status(drift_score, rmse, timestamp) -> str:
    logging.info(f"Computing status at {timestamp} with drift_score={drift_score}, rmse={rmse}")
    if drift_score is not None and drift_score > DRIFT_CRITICAL_THRESHOLD:
        return "DEGRADED"
    if rmse is not None and rmse > RMSE_WARNING_THRESHOLD:
        return "WARNING"
    return "OK"

def extract_drift_score(drift_metrics: dict):
    for metric in drift_metrics.get("metrics", []):
        metric_name = metric.get("metric_name", "")
        if "ValueDrift" in metric_name:
            return metric.get("value")
    return None

def extract_perf_metrics(perf_metrics: dict) -> dict:
    rmse = None
    mae = None
    r2 = None

    for metric in perf_metrics.get("metrics", []):
        metric_name = metric.get("metric_name", "")
        value = metric.get("value")

        if "RMSE" in metric_name:
            rmse = value

        elif "MAE" in metric_name and isinstance(value, dict):
            mae = value.get("mean")

        elif "R2Score" in metric_name:
            r2 = value

    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
    }

def generate_monitoring_metrics(storage: MetricsStorage):

    latest_ref_drift = get_latest_versioned_file(REFERENCE_DRIFT)
    latest_ref_perf = get_latest_versioned_file(REFERENCE_PERF)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    ref_drift = pd.read_csv(latest_ref_drift)
    ref_perf = pd.read_csv(latest_ref_perf)

    preds = pd.read_json(PREDICTION_LOG, lines=True)
    flat_preds = flatten_predictions(preds)

    if flat_preds.empty:
        raise ValueError("No predictions available after flattening prediction logs.")

    model_version = preds["model_version"].iloc[0]
    model_name = preds["model_name"].iloc[0]

    merged = pd.DataFrame()

    if os.path.exists(FEEDBACK_LOG):
        feedbacks = pd.read_json(FEEDBACK_LOG, lines=True)
        flat_feedbacks = flatten_predictions(feedbacks)

        if not flat_feedbacks.empty:
            merged = flat_preds.merge(flat_feedbacks, on="prediction_id", how="left")
            merged = merged[
                [c for c in merged.columns if not c.endswith("_y") or c == "target_y"]
            ]
            merged = merged.rename(columns={"target_x": "prediction", "target_y": "target"})
            merged = merged.dropna(subset=["prediction", "target"]).reset_index(drop=True)
        else:
            logging.warning("Flattened feedbacks dataframe is empty")
    else:
        logging.warning("Feedback log file does not exist")

    drift_report = Report(metrics=[DataDriftPreset(drift_share=0.7)])
    result_drift_report = drift_report.run(
        reference_data=ref_drift[["target"]],
        current_data=flat_preds.drop(columns=["prediction_id"], errors="ignore")[["target"]],
    )

    drift_metrics_raw = result_drift_report.json()

    drift_metrics = json.loads(drift_metrics_raw)

    drift_score = extract_drift_score(drift_metrics)

    metrics = {
        "timestamp": timestamp,
        "model_name": model_name,
        "model_version": str(model_version),
        "drift_score": drift_score,
        "mae": None,
        "rmse": None,
        "r2": None,
    }

    if not merged.empty:
        merged = merged[["target", "prediction"]].dropna().copy()

        definition = DataDefinition(
            regression=[Regression(target="target", prediction="prediction")]
        )
        reference = Dataset.from_pandas(ref_perf, data_definition=definition)
        current = Dataset.from_pandas(merged, data_definition=definition)

        perf_report = Report(metrics=[RegressionPreset()], include_tests=True)
        result_perf_report = perf_report.run(reference_data=reference, current_data=current)

        perf_metrics_raw = result_perf_report.json()

        perf_metrics = json.loads(perf_metrics_raw)

        metrics.update(extract_perf_metrics(perf_metrics))
    else:
        logging.warning("Merged dataframe is empty, performance metrics will remain None")

    status = compute_status(
        metrics.get("drift_score"),
        metrics.get("rmse"),
        timestamp
    )

    metrics["status"] = status

    storage.store_metrics(metrics)