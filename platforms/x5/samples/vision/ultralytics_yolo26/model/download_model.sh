#!/bin/bash

# Download YOLO26 Nano Models (Lightweight for quick start)
# Usage: sh download_model.sh

# Create model directory if not exists
mkdir -p .

# Base URL
BASE_URL="https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_x5/Ultralytics_YOLO_OE_1.2.8/"

# List of models to download (Only n series)
MODELS=(
    "yolo26n_detect_bayese_640x640_nv12.bin"
    "yolo26n_seg_bayese_640x640_nv12.bin"
    "yolo26n_pose_bayese_640x640_nv12.bin"
    "yolo26n_obb_bayese_640x640_nv12.bin"
    "yolo26n_cls_bayese_224x224_nv12.bin"
)

echo "Downloading YOLO26 Nano models..."

for model in "${MODELS[@]}"; do
    if [ ! -f "$model" ]; then
        echo "Downloading $model ..."
        wget -q --show-progress "${BASE_URL}/${model}" -O "$model"
        if [ $? -ne 0 ]; then
            echo "Failed to download $model"
        else
            echo "Successfully downloaded $model"
        fi
    else
        echo "$model already exists, skipping."
    fi
done

echo "Download complete."
