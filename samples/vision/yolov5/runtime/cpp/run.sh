#!/bin/bash
set -e

# Read SOC information
SOC=$(tr 'A-Z' 'a-z' </sys/class/boardinfo/soc_name)
echo "SOC        : $SOC"

# Environment Setup
# (Add any C++ specific environment setup if needed)

# Model Download
MODEL_PATH="../../model/yolov5x_672x672_nv12.hbm"
MODEL_URL="https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_${SOC}/ultralytics_YOLO/yolov5x_672x672_nv12.hbm"

echo "Model path : $MODEL_PATH"

if [[ ! -f "$MODEL_PATH" ]]; then
  echo "Model not found, downloading..."

  mkdir -p "$(dirname "$MODEL_PATH")"

  curl -fL "$MODEL_URL" -o "$MODEL_PATH"

  echo "Model downloaded successfully"
else
  echo "Model already exists, skip download"
fi

# Build
mkdir -p build
cd build
cmake ..
make -j$(nproc)

# Model Execution
./yolov5 \
    --model_path "$MODEL_PATH" \
    --test_img ../../../../../datasets/coco/assets/kite.jpg \
    --label_file ../../../../../datasets/coco/coco_classes.names \
    --score_thres 0.25 \
    --nms_thres 0.45
