#!/bin/bash
set -e

# LaneNet model only supports RDK S100 platform
MODEL_URL="https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/Lanenet/lanenet256x512.hbm"

echo "Model URL  : $MODEL_URL"
echo ""
echo "[NOTE] LaneNet only supports RDK S100. This model is NOT compatible with RDK S600."
echo ""

wget "$MODEL_URL"
