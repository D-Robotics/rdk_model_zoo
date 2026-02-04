#!/bin/bash

# One-click execution script for YOLO26 Sample
# Usage: sh run.sh

# 1. Setup paths
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/../../../../.." && pwd)
MODEL_DIR="$SCRIPT_DIR/../../model"
TEST_DATA_DIR="$PROJECT_ROOT/datasets/coco/assets"
RESULT_DIR="$SCRIPT_DIR/../../test_data"

# 2. Download Models if missing (Default to Nano Detect)
MODEL_FILE="yolo26n_detect_bayese_640x640_nv12.bin"

if [ ! -f "$MODEL_DIR/$MODEL_FILE" ]; then
    echo "Model $MODEL_FILE not found. Downloading..."
    cd "$MODEL_DIR" || exit
    sh download_model.sh
    cd "$SCRIPT_DIR" || exit
fi

# 3. Create Result Directory
mkdir -p "$RESULT_DIR"

# 4. Run Inference (Detect Task)
echo "Running YOLO26n Detect..."
python3 main.py \
    --task detect \
    --model-path "$MODEL_DIR/$MODEL_FILE" \
    --test-img "$TEST_DATA_DIR/bus.jpg" \
    --img-save-path "$RESULT_DIR/result_detect.jpg"

if [ $? -eq 0 ]; then
    echo "Inference finished. Result saved to $RESULT_DIR/result_detect.jpg"
else
    echo "Inference failed."
fi
