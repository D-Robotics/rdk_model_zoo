#!/bin/bash
set -e

SOC=$(tr 'A-Z' 'a-z' </sys/class/boardinfo/soc_name)
MODEL_URL="https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_${SOC}/unetmobilenet/unet_mobilenet_1024x2048_nv12.hbm"

echo "SOC        : $SOC"
echo "Model URL  : $MODEL_URL"

wget "$MODEL_URL"
