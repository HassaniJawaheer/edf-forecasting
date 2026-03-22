"""edf-forecasting
"""

__version__ = "0.1"

import mlflow
import os

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "file:/mlruns"))
