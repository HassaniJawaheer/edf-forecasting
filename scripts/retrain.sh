#!/usr/bin/env bash
set -e

echo " EDF FOrecasting - Manual Retraining"

if ! command -v kedro &> /dev/null; then
    echo "Kedro is not available. Activate the retraining environment first."
    exit 1
first

echo "▶ Starting Kedro training pipeline..."
echo

# Retraining
uv run kedro run --pipeline xgboost_time_series

echo
echo "Retraining finished successfully."
echo "Check MLflow for the new run and model version."