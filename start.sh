#!/bin/bash
set -e

echo "Starting MLFlow server..."
mlflow server \
    --backend-store-uri sqlite:///mlflow.db \
    --host 127.0.0.1 \
    --port 7001 &

# Wait for MLFlow server to be ready
echo "Waiting for MLFlow server to start..."
for i in $(seq 1 30); do
    if curl -s http://127.0.0.1:7001/health > /dev/null 2>&1; then
        echo "MLFlow server is ready!"
        break
    fi
    sleep 1
done

echo "Running train.py to register model..."
python train.py

echo "Starting FastAPI server..."
uvicorn main:app --host 0.0.0.0 --port 8000
