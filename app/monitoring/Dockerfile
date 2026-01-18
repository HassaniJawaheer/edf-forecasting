FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/list/*


ENV PYTHONDONTWRITEBYCODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/list/*

COPY requirements-monitoring.txt .

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements-monitoring.txt

COPY src/monitoring/ monitoring/

CMD ["python", "monitoring/run_monitoring.py"]