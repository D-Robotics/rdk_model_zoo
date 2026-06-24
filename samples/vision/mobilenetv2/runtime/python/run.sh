#!/bin/bash
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

# Environment Setup
#
# Probe by import name; only install pinned fallbacks when the package is not
# importable, so newer pre-installed versions are kept in place.
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

# Model Download
MODEL_PATH="/opt/hobot/model/${MODEL_SOC}/basic/mobilenetv2_224x224_nv12.hbm"
MODEL_URL="https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_${MODEL_SOC}/MobileNet/mobilenetv2_224x224_nv12.hbm"

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
  --test-img ../../test_data/zebra_cls.jpg \
  --label-file ../../test_data/imagenet1000_labels.txt
