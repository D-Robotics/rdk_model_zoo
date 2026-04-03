#!/bin/bash
set -e
python3 -c "import numpy, cv2, scipy" || pip3 install numpy opencv-python scipy

MODEL_PATH="/opt/hobot/model/x5/basic/ConvNeXt_atto_224x224_nv12.bin"
[ ! -f "$MODEL_PATH" ] && MODEL_PATH="../../model/ConvNeXt_atto_224x224_nv12.bin"
[ ! -f "$MODEL_PATH" ] && bash ../../model/download.sh && MODEL_PATH="../../model/ConvNeXt_atto_224x224_nv12.bin"

python3 main.py --model-path "$MODEL_PATH"