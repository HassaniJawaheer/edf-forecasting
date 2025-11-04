import os
import json
from datetime import datetime

LOG_DIR = "src/logs"
PREDICTION_LOG_FILE = os.path.join(LOG_DIR, "predictions.jsonl")
FEEDBACK_LOG_FILE = os.path.join(LOG_DIR, "ground_truth.jsonl")

# Create logs directory if do not exist
os.makedirs(LOG_DIR, exist_ok=True)


def log_predictions(inputs, outputs, model_name, model_version, n_predictions, prediction_id):
    """Add a line in predictions.jsonl"""
    record = {
        "timestamp": datetime.now().isoformat(),
        "model_name": model_name,
        "model_version": model_version,
        "prediction_id": prediction_id,
        "n_predictions": n_predictions,
        "inputs": inputs,
        "outputs": outputs,
        "n_inputs": len(inputs),
        "n_outputs": len(outputs)
    }
    with open(PREDICTION_LOG_FILE, "a") as f:
        f.write(json.dumps(record) + "\n")

def log_feedback(inputs, outputs, prediction_id):
    """Add a lin in grounds_truth.jsonl"""
    record = {
        "timespamp": datetime.now().isoformat(),
        "prediction_id": prediction_id,
        "inputs": inputs,
        "outputs": outputs
    }
    with open(FEEDBACK_LOG_FILE, "a") as f:
        f.write(json.dumps(record) + "\n")