#!/bin/bash

set -e

MODEL_PATH="/opt/hobot/model/x5/basic/FasterNet_S_224x224_nv12.bin"

if [ ! -f "$MODEL_PATH" ]; then
    MODEL_PATH="../../model/FasterNet_S_224x224_nv12.bin"
fi

if [ ! -f "$MODEL_PATH" ]; then
    bash ../../model/download.sh
    MODEL_PATH="../../model/FasterNet_S_224x224_nv12.bin"
fi

python3 main.py --model-path "$MODEL_PATH"
