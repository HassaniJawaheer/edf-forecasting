import os
import json
import pandas as pd
import logging
import glob
import mlflow
from datetime import datetime
from evidently import Dataset, Report, DataDefinition, Regression
from evidently.presets import DataDriftPreset, RegressionPreset
from apscheduler.schedulers.background import BackgroundScheduler
from src.edf_forecasting_api.monitoring.metrics_storage import MetricsStorage

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("edf_forecasting_monitoring")

LOG_DIR = "src/logs"
REFERENCE_DRIFT = "./data/03_primary/eco2mix/definitive/30min/checked/reference/reference_data_drift.csv"
REFERENCE_PERF = "./data/03_primary/eco2mix/definitive/30min/checked/reference/reference_data_perf.csv"
PREDICTION_LOG = os.path.join(LOG_DIR, "predictions.jsonl")
FEEDBACK_LOG = os.path.join(LOG_DIR, "ground_truth.jsonl")
REPORT_DIR = "src/reports"

os.makedirs(REPORT_DIR, exist_ok=True)

def get_latest_versioned_file(base_path: str) -> str:
    pattern = os.path.join(base_path, "*/", os.path.basename(base_path))
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(f"No versioned files found for {base_path}")
    return sorted(matches)[-1]

def flatten_predictions(df):
    rows = []
    for _, row in df.iterrows():
        inputs = row.get("inputs", [])
        outputs = row.get("outputs", [])
        if isinstance(inputs, list) and isinstance(outputs, list):
            for idx, (inp, out) in enumerate(zip(inputs, outputs)):
                first_val = out[0] if isinstance(out, list) else out
                entry = {f"consumption_{i+1}": v for i, v in enumerate(inp)}
                entry.update({
                    "prediction_id": f"{row['prediction_id']}_{idx}",
                    "target": first_val
                })
                rows.append(entry)
    return pd.DataFrame(rows)

def generate_monitoring_reports(storage: MetricsStorage):
    latest_ref_drift = get_latest_versioned_file(REFERENCE_DRIFT)
    latest_ref_perf = get_latest_versioned_file(REFERENCE_PERF)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(REPORT_DIR, timestamp)
    os.makedirs(run_dir, exist_ok=True)

    ref_drift = pd.read_csv(latest_ref_drift)
    ref_perf = pd.read_csv(latest_ref_perf)

    preds = pd.read_json(PREDICTION_LOG, lines=True)
    flat_preds = flatten_predictions(preds)

    model_version = preds["model_version"][0]
    model_name = preds["model_name"][0]

    if os.path.exists(FEEDBACK_LOG):
        feedbacks = pd.read_json(FEEDBACK_LOG, lines=True)
        flat_feedbacks = flatten_predictions(feedbacks)
        merged = flat_preds.merge(flat_feedbacks, on="prediction_id", how="left")
        merged = merged[[c for c in merged.columns if not c.endswith("_y") or c == "target_y"]]
        merged = merged.rename(columns={"target_x": "prediction", "target_y": "target"})
        merged = merged.dropna(subset=["prediction", "target"]).reset_index(drop=True)
    else:
        merged = pd.DataFrame()

    drift_report = Report(metrics=[DataDriftPreset(drift_share=0.7)])
    result_drift_report = drift_report.run(
        reference_data=ref_drift[["target"]],
        current_data=flat_preds.drop(columns=["prediction_id"], errors="ignore")[["target"]]
    )
    drift_report_html = result_drift_report.get_html_str(as_iframe=False)
    drift_report_html_path = os.path.join(run_dir, "data_drift_report.html")
    with open(drift_report_html_path, "w") as f:
        f.write(drift_report_html)

    drift_metrics = json.loads(result_drift_report.json())

    perf_metrics = None
    if not merged.empty:
        merged = merged[["target", "prediction"]].dropna().copy()
        definition = DataDefinition(regression=[Regression(target="target", prediction="prediction")])
        reference = Dataset.from_pandas(ref_perf, data_definition=definition)
        current = Dataset.from_pandas(merged, data_definition=definition)

        perf_report = Report(metrics=[RegressionPreset()], include_tests=True)
        result_perf_report = perf_report.run(reference_data=reference, current_data=current)
        perf_report_html = result_perf_report.get_html_str(as_iframe=False)

        perf_report_html_path = os.path.join(run_dir, "performance_report.html")
        with open(perf_report_html_path, "w") as f:
            f.write(perf_report_html)

        perf_metrics = json.loads(result_perf_report.json())

    metrics = {
        "timestamp": timestamp,
        "model_name": model_name,
        "model_version": str(model_version),
        "drift_report_path": drift_report_html_path,
        "drift_score": next(m["value"] for m in drift_metrics["metrics"] if "ValueDrift" in m["metric_id"])
    }

    if perf_metrics:
        perf_index = {m["metric_id"]: m["value"] for m in perf_metrics["metrics"]}
        metrics.update({
            "rmse": next((v for k, v in perf_index.items() if "RMSE" in k), None),
            "mae": next((v["mean"] for k, v in perf_index.items() if "MAE" in k and isinstance(v, dict)), None),
            "r2": next((v for k, v in perf_index.items() if "R2Score" in k), None),
            "perf_report_path": perf_report_html_path,
        })

    storage.store_metrics(metrics)
    logging.info(f"Performance and Drift report generated at {run_dir}")

    with mlflow.start_run(run_name=f"monitoring_{timestamp}"):
        mlflow.log_artifact(run_dir, artifact_path="monitoring_reports")
        mlflow.log_metrics({
            "drift_score": metrics["drift_score"],
            "rmse": metrics.get("rmse", 0.0),
            "mae": metrics.get("mae", 0.0),
            "r2": metrics.get("r2", 0.0)
        })
        mlflow.set_tags({
            "model_name": model_name,
            "model_version": str(model_version),
            "timestamp": timestamp
        })

def schedule_monitoring(storage):
    scheduler = BackgroundScheduler()
    scheduler.add_job(lambda: generate_monitoring_reports(storage), "interval", seconds=500)
    scheduler.start()
