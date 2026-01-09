#!/bin/bash
set -e

# Read SOC information
SOC=$(tr 'A-Z' 'a-z' </sys/class/boardinfo/soc_name)
echo "SOC        : $SOC"

# Environment Setup
PYTHON_BIN=python3
PIP_BIN=pip3

REQUIREMENTS=(
  "numpy==1.26.4"
  "opencv-python==4.11.0.86"
  "scipy==1.15.3"
  "lap==0.5.12"
  "Cython==3.2.4"
  "cython_bbox==0.1.5"
)

check_and_install() {
  local pkg="$1"
  local name="${pkg%%==*}"
  local version="${pkg##*==}"

  installed_version=$($PIP_BIN show "$name" 2>/dev/null | awk '/^Version:/{print $2}')

  if [[ "$installed_version" == "$version" ]]; then
    echo "$name==$version already installed, skip"
  else
    if [[ -n "$installed_version" ]]; then
      echo "$name version mismatch (installed: $installed_version, need: $version)"
    else
      echo "$name not installed, installing $version"
    fi
    $PIP_BIN install "$name==$version"
  fi
}

for pkg in "${REQUIREMENTS[@]}"; do
  check_and_install "$pkg"
done

# Model Download (YOLOv5x)
MODEL_PATH="../../model/yolov5x_672x672_nv12.hbm"
MODEL_URL="https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_${SOC}/ultralytics_YOLO/yolov5x_672x672_nv12.hbm"

echo "Model path : $MODEL_PATH"

# Function to check if file is valid (larger than 1MB)
is_valid_model() {
    if [[ ! -f "$1" ]]; then return 1; fi
    filesize=$(stat -c%s "$1")
    if (( filesize < 1048576 )); then return 1; fi
    return 0
}

if ! is_valid_model "$MODEL_PATH"; then
  echo "Model not found or invalid, downloading..."
  mkdir -p "$(dirname "$MODEL_PATH")"
  
  # Remove invalid file if exists
  if [[ -f "$MODEL_PATH" ]]; then rm "$MODEL_PATH"; fi
  
  curl -fL "$MODEL_URL" -o "$MODEL_PATH"
  
  if ! is_valid_model "$MODEL_PATH"; then
      echo "Error: Downloaded model seems invalid (too small). Please check network or URL."
      exit 1
  fi
  echo "Model downloaded successfully"
else
  echo "Model already exists and seems valid, skip download"
fi

# Test Data Download
VIDEO_PATH="../../test_data/test_video.mp4"
VIDEO_URL="https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/ByteTrack/track_test.mp4"

if [[ ! -f "$VIDEO_PATH" ]]; then
  echo "Test video not found, downloading..."
  mkdir -p "$(dirname "$VIDEO_PATH")"
  curl -fL "$VIDEO_URL" -o "$VIDEO_PATH"
  echo "Video downloaded successfully"
else
  echo "Video already exists, skip download"
fi

# Model Execution
$PYTHON_BIN main.py \
    --model-path "$MODEL_PATH" \
    --input "$VIDEO_PATH" \
    --output "result.mp4" \
    --priority 0 \
    --bpu-cores 0 \
    --score-thres 0.25 \
    --track-thresh 0.3
