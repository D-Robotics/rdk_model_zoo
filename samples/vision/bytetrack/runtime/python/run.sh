#!/bin/bash
set -e

# Read SOC information
SOC=$(tr 'A-Z' 'a-z' </sys/class/boardinfo/soc_name)
echo "SOC        : $SOC"

# Environment Setup
PYTHON_BIN=python3
PIP_BIN=pip3

REQUIREMENTS=(
  "numpy>=1.26.4"
  "opencv-python>=4.11.0.86"
  "scipy>=1.15.3"
  "lap>=0.4.0"
  "Cython>=3.2.4"
  "cython_bbox>=0.1.5"
)

check_and_install() {
  local pkg="$1"
  echo "Checking dependency: $pkg"
  $PIP_BIN install "$pkg" --quiet
}

for pkg in "${REQUIREMENTS[@]}"; do
  check_and_install "$pkg"
done

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

# Video Download
VIDEO_PATH="../../test_data/track_test.mp4"
VIDEO_URL="https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/ByteTrack/track_test.mp4"

if [[ ! -f "$VIDEO_PATH" ]]; then
  echo "Video not found, downloading..."
  mkdir -p "$(dirname "$VIDEO_PATH")"
  wget -c "$VIDEO_URL" -O "$VIDEO_PATH"
  if [ $? -ne 0 ]; then
    echo "Failed to download video."
    exit 1
  fi
  echo "Video downloaded successfully"
else
  echo "Video already exists, skip download"
fi

# Model Execution
python3 main.py \
    --model-path "$MODEL_PATH" \
    --input "$VIDEO_PATH" \
    --output ../../test_data/result.mp4 \
    --score-thres 0.25 \
    --track-thresh 0.3
