docker run -d \
  --name mlflow-server \
  --network edf-network \
  -p 5000:5000 \
  -e MLFLOW_S3_ENDPOINT_URL=http://minio:9000 \
  -e AWS_ACCESS_KEY_ID=admin \
  -e AWS_SECRET_ACCESS_KEY=password \
  -e AWS_DEFAULT_REGION=us-east-1 \
  edf-mlflow