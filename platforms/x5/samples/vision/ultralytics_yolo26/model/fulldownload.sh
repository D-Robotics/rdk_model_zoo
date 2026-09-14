#!/bin/bash

# Download ALL YOLO26 Models (n, s, m, l, x for all tasks)
# Usage: sh fulldownload.sh

# Create model directory if not exists
mkdir -p .

# Base URL
BASE_URL="https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_x5/Ultralytics_YOLO_OE_1.2.8/"

# List of all models
MODELS=(
    "yolo26n_detect_bayese_640x640_nv12.bin"
    "yolo26s_detect_bayese_640x640_nv12.bin"
    "yolo26m_detect_bayese_640x640_nv12.bin"
    "yolo26l_detect_bayese_640x640_nv12.bin"
    "yolo26x_detect_bayese_640x640_nv12.bin"
    
    "yolo26n_seg_bayese_640x640_nv12.bin"
    "yolo26s_seg_bayese_640x640_nv12.bin"
    "yolo26m_seg_bayese_640x640_nv12.bin"
    "yolo26l_seg_bayese_640x640_nv12.bin"
    "yolo26x_seg_bayese_640x640_nv12.bin"

    "yolo26n_pose_bayese_640x640_nv12.bin"
    "yolo26s_pose_bayese_640x640_nv12.bin"
    "yolo26m_pose_bayese_640x640_nv12.bin"
    "yolo26l_pose_bayese_640x640_nv12.bin"
    "yolo26x_pose_bayese_640x640_nv12.bin"

    "yolo26n_obb_bayese_640x640_nv12.bin"
    "yolo26s_obb_bayese_640x640_nv12.bin"
    "yolo26m_obb_bayese_640x640_nv12.bin"
    "yolo26l_obb_bayese_640x640_nv12.bin"
    "yolo26x_obb_bayese_640x640_nv12.bin"

    "yolo26n_cls_bayese_224x224_nv12.bin"
    "yolo26s_cls_bayese_224x224_nv12.bin"
    "yolo26m_cls_bayese_224x224_nv12.bin"
    "yolo26l_cls_bayese_224x224_nv12.bin"
    "yolo26x_cls_bayese_224x224_nv12.bin"
)

echo "Downloading ALL YOLO26 models..."

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

echo "All downloads complete."
