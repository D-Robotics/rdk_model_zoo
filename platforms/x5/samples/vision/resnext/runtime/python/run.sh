#!/bin/bash
set -e

MODEL_PATH="/opt/hobot/model/x5/basic/ResNeXt50_32x4d_224x224_nv12.bin"
[ ! -f "$MODEL_PATH" ] && MODEL_PATH="../../model/ResNeXt50_32x4d_224x224_nv12.bin"
[ ! -f "$MODEL_PATH" ] && bash ../../model/download.sh && MODEL_PATH="../../model/ResNeXt50_32x4d_224x224_nv12.bin"

python3 main.py --model-path "$MODEL_PATH" "$@"
