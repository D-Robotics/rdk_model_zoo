#!/bin/bash
set -e

# Read SOC information
SOC=$(tr 'A-Z' 'a-z' </sys/class/boardinfo/soc_name)
echo "SOC        : $SOC"

# S600 platform check
if [[ "$SOC" == "s600" ]]; then
  echo "[ERROR] YOLOe11-Seg is not supported on S600 platform. Exiting."
  exit 1
fi

# Environment Setup
PYTHON_BIN=python3
PIP_BIN=pip3

REQUIREMENTS=(
  "numpy==1.26.4"
  "opencv-python==4.11.0.86"
  "scipy==1.15.3"
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

# Model Download
MODEL_PATH="/opt/hobot/model/${SOC}/basic/yoloe_11s_seg_pf_nashe_640x640_nv12.hbm"
MODEL_URL="https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/ultralytics_YOLO/yoloe_11s_seg_pf_nashe_640x640_nv12.hbm"

echo "Model path : $MODEL_PATH"

if [[ ! -f "$MODEL_PATH" ]]; then
  echo "Model not found, downloading..."

  mkdir -p "$(dirname "$MODEL_PATH")"

  curl -fL "$MODEL_URL" -o "$MODEL_PATH"

  echo "Model downloaded successfully"
else
  echo "Model already exists, skip download"
fi

# Model Execution
python main.py \
    --model-path /opt/hobot/model/$SOC/basic/yoloe_11s_seg_pf_nashe_640x640_nv12.hbm \
    --test-img ../../test_data/office_desk.jpg \
    --label-file ../../test_data/coco_extended.names \
    --img-save-path result.jpg \
    --priority 0 \
    --bpu-cores 0 \
    --nms-thres 0.7 \
    --score-thres 0.25
