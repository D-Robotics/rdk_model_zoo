#!/bin/bash
set -e

SOC=$(tr 'A-Z' 'a-z' </sys/class/boardinfo/soc_name)

# S600 platform check — YOLOe11-Seg is not supported on S600
if [[ "$SOC" == "s600" ]]; then
  echo "[ERROR] YOLOe11-Seg is not supported on S600 platform. Exiting."
  exit 1
fi

MODEL_URL="https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/ultralytics_YOLO/yoloe_11s_seg_pf_nashe_640x640_nv12.hbm"

echo "SOC        : $SOC"
echo "Model URL  : $MODEL_URL"

wget "$MODEL_URL"
