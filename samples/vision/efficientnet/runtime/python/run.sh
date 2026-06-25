#!/usr/bin/env bash
set -e

# Resolve target SoC. Only s100 and s600 prebuilt HBM files are published;
# anything else (s100p, (null), unknown) falls back to the S100 build.
SOC_RAW=$(cat /sys/class/boardinfo/soc_name 2>/dev/null | tr 'A-Z' 'a-z' | tr -d '()' | xargs)
SOC="${SOC_RAW:-s100}"
case "$SOC" in
  s600) MODEL_SOC="s600" ;;
  *)    MODEL_SOC="s100" ;;
esac

echo "SOC           : $SOC"
echo "Model variant : rdk_${MODEL_SOC}"

# Model Download — download to the system model directory so that main.py
# (with its default SOC-aware model path) can find it.
MODEL_PATH="/opt/hobot/model/${MODEL_SOC}/basic/efficientnet_lite0_224x224_nv12.hbm"
MODEL_URL="https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_${MODEL_SOC}/EfficientNet/efficientnet_lite0_224x224_nv12.hbm"

echo "Model path : $MODEL_PATH"

if [[ ! -f "$MODEL_PATH" ]]; then
  echo "Model not found, downloading..."

  mkdir -p "$(dirname "$MODEL_PATH")"

  if command -v wget &>/dev/null; then
    wget -q "$MODEL_URL" -O "$MODEL_PATH"
  elif command -v curl &>/dev/null; then
    curl -fL "$MODEL_URL" -o "$MODEL_PATH"
  else
    echo "ERROR: neither wget nor curl found" >&2
    exit 1
  fi

  echo "Model downloaded successfully"
else
  echo "Model already exists, skip download"
fi

# Model Execution
python3 main.py \
  --model-path "$MODEL_PATH" \
  --test-img "../../test_data/Scottish_deerhound.JPEG" \
  --label-file "../../test_data/imagenet_classes.names" \
  --top-k 5 \
  "$@"