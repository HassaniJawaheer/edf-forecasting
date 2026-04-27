import os
import sqlite3
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

METRICS_DB = os.getenv("METRICS_DB", "src/reports/db/metrics.db")
ALERTS_DB = os.getenv("ALERTS_DB", "src/reports/db/alerts.db")


def init_alerts_db():
    conn = sqlite3.connect(ALERTS_DB)
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS model_alerts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        alert_timestamp TEXT,
        metric_timestamp TEXT,
        model_name TEXT,
        model_version TEXT,
        mae REAL,
        rmse REAL,
        r2 REAL,
        drift_score REAL,
        status TEXT
    )
    """)

    conn.commit()
    conn.close()


def get_latest_metrics():
    conn = sqlite3.connect(METRICS_DB)
    cursor = conn.cursor()

    row = cursor.execute("""
        SELECT
            timestamp,
            model_name,
            model_version,
            mae,
            rmse,
            r2,
            drift_score,
            status
        FROM performance_metrics
        ORDER BY timestamp DESC
        LIMIT 1
    """).fetchone()

    conn.close()
    return row


def store_alert(row):
    (
        metric_timestamp,
        model_name,
        model_version,
        mae,
        rmse,
        r2,
        drift_score,
        status,
    ) = row

    alert_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    conn = sqlite3.connect(ALERTS_DB)
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO model_alerts (
            alert_timestamp,
            metric_timestamp,
            model_name,
            model_version,
            mae,
            rmse,
            r2,
            drift_score,
            status
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        alert_timestamp,
        metric_timestamp,
        model_name,
        model_version,
        mae,
        rmse,
        r2,
        drift_score,
        status,
    ))

    conn.commit()
    conn.close()

    logging.warning(
        f"[MODEL ALERT] status={status}, model={model_name}, "
        f"version={model_version}, rmse={rmse}, drift_score={drift_score}"
    )


def check_metrics():
    init_alerts_db()

    row = get_latest_metrics()

    if row is None:
        logging.info("No metrics found in performance_metrics.")
        return

    status = row[-1]

    logging.info(f"Latest model status: {status}")

    if status != "OK":
        store_alert(row)
    else:
        logging.info("No alert triggered.")


if __name__ == "__main__":
    check_metrics()