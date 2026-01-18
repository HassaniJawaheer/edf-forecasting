import sqlite3
import logging
from tabulate import tabulate

DB_PATH = "src/reports/db/metrics.db"

def view_metrics(limit=10):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT timestamp, model_name, model_version, mae, rmse, r2, drift_score, drift_report_path, perf_report_path
        FROM performance_metrics
        ORDER BY timestamp DESC
        LIMIT ?
    """, (limit,))
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        logging.info("No metrics found in the database")
        return
    
    headers = ["Timestamp", "Model", "Version", "MAE", "RMSE", "r2", "Drift", "Drift report", "Perf report"]
    print(tabulate(rows, headers=headers))

if __name__ == "__main__":
    view_metrics(limit=20)
    