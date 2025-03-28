#!/bin/bash

# API service directory
SERVICE_DIR="/home/azureuser/imageenhancer/src"

# Socket path within service directory
SOCKET_PATH="$SERVICE_DIR/imageenhancer.sock"

# Calculate optimal number of workers based on CPU cores
# Using number of CPU cores minus 1 for optimal performance, minimum of 1
WORKERS=$(( $(nproc) - 1 ))
if [ "$WORKERS" -lt 1 ]; then
    WORKERS=1
fi

# Activate the environment
source /home/azureuser/imageenhancer/src/.venv/bin/activate
#export ACCELERATE_USE_MULTI_GPU=true
#export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Remove existing socket if it exists
if [ -e "$SOCKET_PATH" ]; then
    rm "$SOCKET_PATH"
fi

# Navigate to service directory
cd "$SERVICE_DIR"
echo "Starting Image Enhancer API"
echo "Socket Path: $SOCKET_PATH"
echo "Working Directory: $SERVICE_DIR"
echo "Recheck Working Directory: $(pwd)"

# Start the FastAPI service using the conda environment's uvicorn
#python /home/azureuser/imageenhancer/src/api.py
# Added --timeout 300 for longer running ML tasks
python -m uvicorn api:app  \
   --uds "$SOCKET_PATH" \
#    --workers $WORKERS
    # --timeout-keep-alive 300 \
    # --log-level info \
    # --proxy-headers \
    # --limit-concurrency 1000