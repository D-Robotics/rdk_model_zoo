#!/bin/bash
MODEL_NAME="mobilenetv1_224x224_nv12.bin"
DOWNLOAD_URL="https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_x5/mobilenetv1_224x224_nv12.bin"
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
if [ ! -f "$DIR/$MODEL_NAME" ]; then
    wget -c "$DOWNLOAD_URL" -P "$DIR"
fi