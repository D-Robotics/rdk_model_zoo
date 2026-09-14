#!/bin/bash
set -e

BASE_URL="https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_x5/clip"

cd "$(dirname "$0")"
for MODEL_NAME in img_encoder.bin text_encoder.onnx; do
    if [ -f "$MODEL_NAME" ]; then
        echo "$MODEL_NAME already exists."
        continue
    fi
    TMP_NAME="${MODEL_NAME}.tmp"
    rm -f "$TMP_NAME"
    wget "${BASE_URL}/${MODEL_NAME}" -O "$TMP_NAME"
    mv "$TMP_NAME" "$MODEL_NAME"
done
