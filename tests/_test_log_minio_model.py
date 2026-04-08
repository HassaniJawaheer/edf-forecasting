import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("debug-minio")

X, y = load_iris(return_X_y=True)
model = RandomForestClassifier(n_estimators=5, random_state=42)
model.fit(X, y)

with mlflow.start_run(run_name="test-log-minio-model"):
    mlflow.log_param("model_type", "RandomForestClassifier")
    mlflow.sklearn.log_model(model, artifact_path="model")