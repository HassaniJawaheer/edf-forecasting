import os
import pandas as pd
import logging
from evidently import Dataset, Report, DataDefinition, Regression
from evidently.presets import DataDriftPreset, RegressionPreset
from apscheduler.schedulers.background import BackgroundScheduler


LOG_DIR = "src/logs"
REFERENCE_DRIFT = "./data/03_primary/eco2mix/definitive/30min/checked/reference/reference_data_drift.csv"
REFERENCE_PERF = "./data/03_primary/eco2mix/definitive/30min/checked/reference/reference_data_perf.csv"
PREDICTION_LOG = os.path.join(LOG_DIR, "predictions.jsonl")
FEEDBACK_LOG = os.path.join(LOG_DIR, "ground_truth.jsonl")
REPORT_DIR = "src/reports"
os.makedirs(REPORT_DIR, exist_ok=True)


import glob

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

def generate_monitoring_reports():
    latest_ref_drift = get_latest_versioned_file(REFERENCE_DRIFT)
    latest_ref_perf = get_latest_versioned_file(REFERENCE_PERF)

    ref_drift = pd.read_csv(latest_ref_drift)
    ref_perf = pd.read_csv(latest_ref_perf)

    preds = pd.read_json(PREDICTION_LOG, lines=True)
    flat_preds = flatten_predictions(preds)

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
    with open(os.path.join(REPORT_DIR, "data_drift_report.html"), "w") as f:
        f.write(drift_report_html)
    logging.info(f"Data drift report generated at src/reports/data_drift_report.html")

    if not merged.empty:
        merged = merged[["target", "prediction"]].dropna().copy()

        definition = DataDefinition(
            regression=[Regression(target="target", prediction="prediction")]
        )

        reference = Dataset.from_pandas(ref_perf, data_definition=definition)
        current = Dataset.from_pandas(merged, data_definition=definition)

        perf_report = Report(metrics=[RegressionPreset()], include_tests=True)
        result_perf_report = perf_report.run(reference_data=reference, current_data=current)
        perf_report_html = result_perf_report.get_html_str(as_iframe=False)
        with open(os.path.join(REPORT_DIR, "performance_report.html"), "w") as f:
            f.write(perf_report_html)
        logging.info(f"Performance report generated at src/reports/performance_report.html")

def schedule_monitoring():
    scheduler = BackgroundScheduler()
    scheduler.add_job(generate_monitoring_reports, "interval", seconds=60)
    scheduler.start()