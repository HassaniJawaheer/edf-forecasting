import sqlite3
import os

class MetricsStorage:
    def __init__(self, db_path="src/reports/db/metrics.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Create the SQLite table if it doesn't exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS performance_metrics (
            timestamp TEXT PRIMARY KEY,
            model_name TEXT,
            model_version TEXT,
            mae REAL,
            rmse REAL,
            r2 REAL,
            drift_score REAL,
            drift_report_path TEXT,
            perf_report_path TEXT
        )
        """)
        conn.commit()
        conn.close()

    def store_metrics(self, metrics: dict):
        """Insert a new metrics record."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
        INSERT INTO performance_metrics (timestamp, model_name, model_version, mae, rmse, r2, drift_score, drift_report_path, perf_report_path)
        VALUES (:timestamp, :model_name, :model_version, :mae, :rmse, :r2, :drift_score, :drift_report_path, :perf_report_path)
        """, metrics)
        conn.commit()
        conn.close()
