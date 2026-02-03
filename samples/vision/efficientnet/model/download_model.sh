#!/bin/bash

# Array of EfficientNet-Lite model files and their resolutions
models=(
  "efficientnet_lite0_224x224_nv12.hbm"
  "efficientnet_lite1_240x240_nv12.hbm"
  "efficientnet_lite2_260x260_nv12.hbm"
  "efficientnet_lite3_300x300_nv12.hbm"
  "efficientnet_lite4_380x380_nv12.hbm"
)

base_url="https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/EfficientNet"

for model_file in "${models[@]}"; do
  model_url="${base_url}/${model_file}"
  
  if [ -f "$model_file" ]; then
    echo "Model $model_file already exists, skipping."
  else
    echo "Downloading $model_file..."
    wget -q --show-progress -O "$model_file" "$model_url"
    if [ $? -ne 0 ]; then
      echo "Failed to download $model_file."
      # Continue to next model instead of exiting immediately
    else
      echo "Successfully downloaded $model_file."
    fi
  fi
done
