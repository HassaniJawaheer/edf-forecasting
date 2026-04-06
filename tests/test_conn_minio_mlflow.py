import os
from pathlib import Path

import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("debug-minio")

with mlflow.start_run(run_name="test-conn-mlflow-minio") as run:
    mlflow.log_param("param", 123)
    mlflow.log_metric("metric", 0.99)

    artifact_file = Path("artifact_test.txt")
    artifact_file.write_text("Tests du stockage des artifacts de mlflow vers minio", encoding="utf-8")

    mlflow.log_artifact(str(artifact_file))

    print("Run ID:", run.info.run_id)
    print("Done")