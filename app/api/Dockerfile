FROM python:3.11-slim

ENV PYTHONDONTWRITEBYCODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app
ENV PYTHONPATH=/app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-api.txt .

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements-api.txt

COPY src/edf_forecasting_api/ src/edf_forecasting_api/

EXPOSE 8000

CMD ["uvicorn", "src.edf_forecasting_api.main:app", "--host", "0.0.0.0", "--port", "8000" ]
