docker run -d \
  --name mlflow-server \
  --network edf-forecasting \
  -p 5000:5000 \
  --env-file .env \
  -v mlflow-data:/mlflow/data \
  edf-mlflow