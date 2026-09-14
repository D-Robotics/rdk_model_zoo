#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

SMALL_MODEL="MobileNetV4_conv_small_224x224_nv12.bin"
SMALL_URL="https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_x5/MobileNetV4_conv_small_224x224_nv12.bin"
MEDIUM_MODEL="MobileNetV4_conv_medium_224x224_nv12.bin"
MEDIUM_URL="https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_x5/MobileNetV4_conv_medium_224x224_nv12.bin"

if [ ! -f "$DIR/$SMALL_MODEL" ]; then
    wget -c "$SMALL_URL" -P "$DIR"
fi

if [ ! -f "$DIR/$MEDIUM_MODEL" ]; then
    wget -c "$MEDIUM_URL" -P "$DIR"
fi
