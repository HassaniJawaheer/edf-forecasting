FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /mlflow

COPY requirements-mlflow.txt .

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements-mlflow.txt

EXPOSE 5000

CMD ["mlflow", "server", "--backend-store-uri", "sqlite:////mlflow/data/mlflow.db", "--artifacts-destination", "s3://mlflow-artifacts", "--host", "0.0.0.0", "--port", "5000"]