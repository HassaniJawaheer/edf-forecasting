from src.monitoring.ml_monitoring import generate_monitoring_metrics
from src.monitoring.metrics_storage import MetricsStorage
from apscheduler.schedulers.blocking import BlockingScheduler
import os
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logging.getLogger("apscheduler").setLevel(logging.WARNING)

METRICS_DB = os.getenv("METRICS_DB", "src/reports/db/metrics.db")
INTERVAL_SECONDS = int(os.getenv("MONITORING_INTERVAL_SECONDS", "15"))

def run_job():
    logging.info("Starting monitoring job")
    storage = MetricsStorage(METRICS_DB)
    generate_monitoring_metrics(storage)
    logging.info("Monitoring job finished")

if __name__ == "__main__":
    scheduler = BlockingScheduler()

    run_job()

    scheduler.add_job(run_job, "interval", seconds=INTERVAL_SECONDS)

    logging.info(f"Scheduler started (interval={INTERVAL_SECONDS}s)")
    scheduler.start()