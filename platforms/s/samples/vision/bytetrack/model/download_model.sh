#!/bin/bash

SOC="${1:-${SOC:-s100}}"
model_file="yolov5x_672x672_nv12.hbm"
model_url="https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_${SOC}/ultralytics_YOLO/${model_file}"
model_dir="./${SOC}"
model_path="${model_dir}/${model_file}"

mkdir -p "${model_dir}"

if [ -f "${model_path}" ]; then
  echo "Model ${model_path} already exists."
else
  echo "Downloading ${model_path}..."
  wget -O "${model_path}" "$model_url"
  if [ $? -ne 0 ]; then
    echo "Failed to download model."
    exit 1
  fi
  echo "Download successful."
fi
