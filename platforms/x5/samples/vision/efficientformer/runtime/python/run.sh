#!/bin/bash
set -e

MODEL_PATH="/opt/hobot/model/x5/basic/EfficientFormer_l3_224x224_nv12.bin"
[ ! -f "$MODEL_PATH" ] && MODEL_PATH="../../model/EfficientFormer_l3_224x224_nv12.bin"
[ ! -f "$MODEL_PATH" ] && bash ../../model/download.sh && MODEL_PATH="../../model/EfficientFormer_l3_224x224_nv12.bin"

python3 main.py --model-path "$MODEL_PATH"
