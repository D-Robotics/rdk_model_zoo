#!/bin/bash

set -e

MODEL_PATH="../../model/ResNeXt50_32x4d_224x224_nv12.bin"

if [ ! -f "$MODEL_PATH" ]; then
    bash ../../model/download.sh
fi

python3 main.py --model-path "$MODEL_PATH"
