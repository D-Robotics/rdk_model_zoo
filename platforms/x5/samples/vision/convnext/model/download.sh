#!/bin/bash
# Copyright (c) 2025 D-Robotics Corporation
# Standard ConvNeXt Download Script

MODEL_NAME="ConvNeXt_atto_224x224_nv12.bin"
DOWNLOAD_URL="https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_x5/ConvNeXt_atto_224x224_nv12.bin"

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

if [ -f "$DIR/$MODEL_NAME" ]; then
    echo "Model $MODEL_NAME already exists in $DIR. Skipping download."
else
    echo "Downloading $MODEL_NAME to $DIR..."
    wget -c "$DOWNLOAD_URL" -P "$DIR"
fi