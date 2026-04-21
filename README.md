# edf-forecasting

[![CI EDF Forecasting](https://github.com/HassaniJawaheer/edf-forecasting/actions/workflows/ci.yml/badge.svg)](https://github.com/HassaniJawaheer/edf-forecasting/actions/workflows/ci.yml)

> **MLOps pipeline for forecasting electricity consumption in France**

## **Description**
This project aims to build an Electricity Consumption Forecasting API using EDF (Electricité de France) data.

## **Architecture**
The system is composed of several Dockerized services, each with a specific role. A **FastAPI-based** inference API serves predictions and logs both prediction data and user feedback. Model performance and system metrics are monitored using **Prometheus** and **Grafana**, while model performance and data drift are tracked using **Evidently AI**, executed periodically with a scheduler. Model training is handled by an offline training pipeline (with **Kedro**) running outside of Docker. **MLflow** is used for experiment tracking and model registry, and the API automatically loads the latest production model from MLflow. All artifacts, including trained models, are stored in **MinIO**.

## **Installation**

### Prerequisites

Tested with:
* **Python 3.12**
* **Git 2.39.5**
* **uv 0.6.12**
* **Docker 28.2.1**

### Docker installation

Install Docker Engine :

```bash
sudo apt update
sudo apt install -y ca-certificates curl gnupg
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/debian/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
  https://download.docker.com/linux/debian \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io
```
Add user to docker group:

```bash
sudo usermod -aG docker $USER
newgrp docker
```
Verification :

```bash
docker run hello-world
```

### `uv` installation

Execute the command below:

```bash
curl -Ls https://astral.sh/uv/install.sh | sh
```

Reload the shell:

```bash
source ~/.bashrc
```

Verify installation:

```bash
uv --version
```

**See the installation note below**

> [https://docs.astral.sh/uv/getting-started/installation/](https://docs.astral.sh/uv/getting-started/installation/)

### **Copy the projet**

Clone the Git repository :

```bash
git clone https://github.com/HassaniJawaheer/edf-forecasting.git
cd edf-forecasting
```

### **Environment and dependencies setup**

Set up and synchronize the environment :

```bash
uv sync
```
Run a simple kedro pipeline:

```bash
uv run kedro run --pipeline=hello_mlflow
```

### Run the Docker services

First, create a Docker network named `edf-forecasting`.

```bash
docker network create edf-forecasting
```

#### Start MinIO

```bash
docker run -d \
  --name minio \
  --network edf-forecasting \
  -p 9000:9000 \
  -p 9001:9001 \
  --env-file .env \
  -v minio-data:/data \
  minio/minio server /data --console-address ":9001"
```

#### Build MLflow image

```bash
docker build -t edf-forecasting-mlflow -f app/mlflow/Dockerfile .
```

#### Start MLflow container

```bash
docker run -d \
  --name mlflow-server \
  --network edf-forecasting \
  -p 5000:5000 \
  --env-file .env \
  -v $(pwd)/mlflow-server/mlflow-data:/mlflow/data \
  edf-forecasting-mlflow
```

#### Build API image

```bash
docker build -t edf-forecasting-api -f app/api/Dockerfile .
```

#### Build monitoring image

```bash
docker build -t edf-forecasting-monitoring -f app/monitoring/Dockerfile .
```

### Start monitoring container

```bash
docker run -d \
  --name edf-forecasting-monitoring \
  --network edf-forecasting \
  --env-file .env \
  -v $(pwd)/data/03_primary/eco2mix:/app/data/03_primary/eco2mix \
  -v $(pwd)/src/logs:/app/src/logs \
  -v $(pwd)/src/reports/db:/app/src/reports/db \
  edf-forecasting-monitoring
```

### Start Prometheus

```bash
docker run -d \
  --name prometheus \
  --network edf-forecasting \
  -p 9090:9090 \
  -v $(pwd)/app/prometheus/prometheus.yml:/etc/prometheus/prometheus.yml \
  -v prometheus-data:/prometheus \
  prom/prometheus
```

### Start Grafana

```bash
docker run -d \
  --name grafana \
  --network edf-forecasting \
  -p 3000:3000 \
  grafana/grafana
```

### MLflow / Kedro note

`conf/local` has priority over `conf/base`.

If `conf/local/mlflow.yml` contains `null`, it can override the correct value from `conf/base` and fall back to the default `mlruns` directory.

Always define:

```yaml
server:
  mlflow_tracking_uri: http://localhost:5000
```

in `conf/local/mlflow.yml`.

### Monitoring overview

The project uses two monitoring layers:

* **Technical monitoring** with Prometheus and Grafana
* **Model monitoring** with Evidently AI and SQLite

## **Run the training pipeline**

```bash
uv run kedro run --pipeline=xgboost_time_series
```

### Check experiments in MLflow

Open MLflow UI:

[http://localhost:5000](http://localhost:5000)

* Go to **Experiments**
* Open `edf_forecasting`
* Check that the run completed successfully

### Promote the model to Production

* Go to **Models**
* Select `timeseries_xgboost_30min`
* Open the latest version
* Set stage to **Production**

## **Start the API container**

Once the model is in Production, start the API:

```bash
docker run -d \
  --name edf-forecasting-api \
  --network edf-forecasting \
  -p 8000:8000 \
  --env-file .env \
  -v $(pwd)/src/logs:/app/src/logs \
  edf-forecasting-api
```

## **Test the API**

You can send a request using `curl`:

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "features": [
         [
           74494.0, 73481.0, 71506.0, 71505.0, 71134.0, 70856.0, 68840.0, 67315.0,
           65749.0, 64838.0, 64041.0, 64379.0, 64210.0, 64469.0, 64437.0, 64559.0,
           64785.0, 64281.0, 64292.0, 64862.0, 65353.0, 65879.0, 66180.0, 66643.0,
           66901.0, 67719.0, 68547.0, 66745.0, 65090.0, 63891.0, 62228.0, 61554.0,
           61263.0, 61469.0, 62443.0, 65700.0, 68890.0, 70497.0, 71766.0, 72562.0,
           72184.0, 71493.0, 70440.0, 69167.0, 68044.0, 68829.0, 71485.0, 70639.0
         ]
       ],
       "n_predictions": 48
     }'
```

### Parameters

* **features**: input time series
  Each list represents electricity consumption values over one day, with a 30-minute frequency (48 values from 00:00 to 23:30).

* **n_predictions**: number of future values to predict
  For example:

  * `1` → predict the next time step (next 30 minutes)
  * `48` → predict the next full day

You can send:

* one day of data (like above)
* multiple days (multiple lists inside `features`)
