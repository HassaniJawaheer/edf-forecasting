#!/usr/bin/env bash
set -e

HOST="127.0.0.1"
PORT="5000"
BACKEND_URI="mlruns"

echo "▶ Starting MLflow Tracking Server (via uv)"
echo "▶ MLflow UI will be available at:"
echo "   http://${HOST}:${PORT}"
echo

uv run mlflow ui \
  --backend-store-uri "${BACKEND_URI}" \
  --host "${HOST}" \
  --port "${PORT}"
