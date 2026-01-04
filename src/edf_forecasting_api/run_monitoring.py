from edf_forecasting_api.monitoring.ml_monitoring import schedule_monitoring
from src.edf_forecasting_api.monitoring.metrics_storage import MetricsStorage
import os
METRICS_DB = os.getenv("METRICS_DB","src/reports/db/metrics.db")

storage = MetricsStorage(METRICS_DB)
schedule_monitoring(storage)
