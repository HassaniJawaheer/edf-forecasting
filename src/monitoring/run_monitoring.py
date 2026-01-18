from monitoring.ml_monitoring import generate_monitoring_reports
from monitoring.metrics_storage import MetricsStorage
import os

METRICS_DB = os.getenv("METRICS_DB","src/reports/db/metrics.db")
if __name__ == "__main__":
    storage = MetricsStorage(METRICS_DB)
    generate_monitoring_reports(storage)
