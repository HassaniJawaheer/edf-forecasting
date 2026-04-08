"""edf-forecasting
"""

__version__ = "0.1"

import mlflow
import os

tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(tracking_uri)
