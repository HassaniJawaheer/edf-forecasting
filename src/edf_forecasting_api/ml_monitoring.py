import os
import pandas as pd
import logging
from evidently import Report
from evidently.presets import DataDriftPreset, RegressionPreset
from apscheduler.schedulers.background import BackgroundScheduler

LOG_DIR = "src/logs"
REFERENCE_DATA = "./data/03_primary/eco2mix/definitive/30min/checked/reference/data.csv"
PREDICTION_LOG = os.path.join(LOG_DIR, "predictions.jsonl")
FEEDBACK_LOG = os.path.join(LOG_DIR, "ground_truth.jsonl")
REPORT_DIR = "src/reports"
os.makedirs(REPORT_DIR, exist_ok=True)

def generate_monitoring_reports():
    if not os.path.exists(PREDICTION_LOG):
       raise FileNotFoundError("Log file not found.")
    if not os.path.exists(REFERENCE_DATA):
        raise FileNotFoundError("Reference file data not found.")

    preds = pd.read_json(PREDICTION_LOG, lines=True)
    ref = pd.read_csv(REFERENCE_DATA, header=None)

    ref["target"] = ref.iloc[:, -1]
    ref["first_pred"] = None

    for i, row in preds.iterrows():
        out = row["output"]
        if isinstance(out, list):
            if len(out) > 0:
                if isinstance(out[0], list):
                    preds.at[i, "first_pred"] = out[0][0]
                else:
                    preds.at[i, "first_pred"] = out[0]

    # Feedbacks
    if os.path.exists(FEEDBACK_LOG):
        feedbacks = pd.read_json(FEEDBACK_LOG, lines=True)
        merged = preds.merge(feedbacks, on="prediction_id", how="left")
        merged = merged.drop(columns=["timestamp_x", "timestamp_y"], errors="ignore")
    else:
        merged = pd.DataFrame()

    # Drift report
    drift_report = Report(metrics=[DataDriftPreset()])
    drift_report.run(reference_data=ref[["target"]], current_data=preds[["first_pred"]])
    drift_report.save_html(os.path.join(REPORT_DIR, "data_drift_report.html"))
    logging.info("Derive report created at reports/data_drift_report.html")

    # Performance report
    if not merged.empty:
        merged["true_output"] = merged["true_output"].apply(lambda x: x[0] if isinstance(x, list) else x)
        merged = merged.rename(columns={"true_output": "target", "first_pred": "prediction"})
        perf_report = Report(metrics=[RegressionPreset()])
        perf_report.run(reference_data=merged[["target", "prediction"]],
                    current_data=merged[["target", "prediction"]])
        perf_report.save_html(os.path.join(REPORT_DIR, "performance_report.html"))
        print("Performance report created: reports/performance_report.html")

def schedule_monitoring():
    schduler = BackgroundScheduler()
    schduler.add_job(generate_monitoring_reports, "interval", week=1)