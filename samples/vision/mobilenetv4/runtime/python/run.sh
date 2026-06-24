#!/bin/bash
set -e

# Resolve target SoC. s600 has its own published HBM; anything else (s100,
# s100p, (null), unknown) uses the S100 build.
SOC_RAW=$(cat /sys/class/boardinfo/soc_name 2>/dev/null | tr 'A-Z' 'a-z' | tr -d '()' | xargs)
SOC="${SOC_RAW:-s100}"
case "$SOC" in
  s600) MODEL_SOC="s600" ;;
  *)    MODEL_SOC="s100" ;;
esac

echo "SOC           : $SOC"
echo "Model variant : rdk_${MODEL_SOC}"

# Environment Setup
PYTHON_BIN=python3
PIP_BIN=pip3

REQUIREMENTS=(
  "numpy:numpy==1.26.4"
  "cv2:opencv-python==4.11.0.86"
  "scipy:scipy==1.15.3"
)

check_and_install() {
  local entry="$1"
  local import_name="${entry%%:*}"
  local pkg_spec="${entry#*:}"
  local pip_name="${pkg_spec%%==*}"

  if $PYTHON_BIN -c "import ${import_name}" >/dev/null 2>&1; then
    local ver
    ver=$($PYTHON_BIN -c "import ${import_name} as m; print(getattr(m, '__version__', 'unknown'))" 2>/dev/null || echo unknown)
    echo "${pip_name} already importable (version: ${ver}), skip"
  else
    echo "${pip_name} not installed, installing fallback ${pkg_spec}"
    $PIP_BIN install "${pkg_spec}" --break-system-packages
  fi
}

for entry in "${REQUIREMENTS[@]}"; do
  check_and_install "$entry"
done

# Model Download (small variant by default)
MODEL_VARIANT="${1:-small}"
shift || true

if [[ "$MODEL_VARIANT" == "medium" ]]; then
  MODEL_NAME="mobilenetv4_medium_256x256_nv12"
else
  MODEL_NAME="mobilenetv4_small_224x224_nv12"
fi

MODEL_PATH="../../model/${MODEL_SOC}/${MODEL_NAME}.hbm"

echo "Model path : $MODEL_PATH"

if [[ ! -f "$MODEL_PATH" ]]; then
  echo "Model not found, downloading to sample-local model directory..."
  (cd ../../model && bash download_model.sh)
else
  echo "Model already exists, skip download"
fi

# Model Execution
python3 main.py \
  --model-variant "$MODEL_VARIANT" \
  --model-path "$MODEL_PATH" \
  --test-img ../../test_data/zebra_cls.jpg \
  --label-file ../../test_data/imagenet_classes.names \
  --top-k 5 \
  "$@"
