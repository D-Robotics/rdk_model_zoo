#!/bin/bash
set -e

SOC=$(tr 'A-Z' 'a-z' </sys/class/boardinfo/soc_name)
MODEL_URL="https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_${SOC}/MobileNet/mobilenetv2_224x224_nv12.hbm"

echo "SOC        : $SOC"
echo "Model URL  : $MODEL_URL"

wget "$MODEL_URL"
