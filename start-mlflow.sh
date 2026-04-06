docker run -d \
  --name mlflow-server \
  --network edf-forecasting \
  -p 5000:5000 \
  --env-file .env \
  -v $(pwd)/mlflow-server/mlflow-data:/mlflow/data \
  edf-forecasting-mlflow