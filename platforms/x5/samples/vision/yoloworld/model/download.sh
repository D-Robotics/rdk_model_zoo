#!/bin/bash
set -e

MODEL_NAME="yolo_world.bin"
MODEL_URL="https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_x5/yolo_world/${MODEL_NAME}"

cd "$(dirname "$0")"
if [ -f "$MODEL_NAME" ]; then
    echo "$MODEL_NAME already exists."
    exit 0
fi

TMP_NAME="${MODEL_NAME}.tmp"
rm -f "$TMP_NAME"
wget "$MODEL_URL" -O "$TMP_NAME"
mv "$TMP_NAME" "$MODEL_NAME"
