#!/bin/bash

# Download EfficientNet-Lite0 model
model_file="efficientnet_lite0_224x224_nv12.hbm"
model_url="https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/EfficientNet/${model_file}"

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
