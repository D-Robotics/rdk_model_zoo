#!/bin/bash

# Default to S100 for manual download script, or use env var
SOC="${SOC:-s100}"
model_file="yolov5x_672x672_nv12.hbm"
model_url="https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_${SOC}/ultralytics_YOLO/${model_file}"

if [ -f "$model_file" ]; then
  echo "Model $model_file already exists."
else
  echo "Downloading $model_file..."
  wget -O "$model_file" "$model_url"
  if [ $? -ne 0 ]; then
    echo "Failed to download model."
    exit 1
  fi
  echo "Download successful."
fi