#!/bin/bash
set -e

# Read SOC information (strip "(null)" / whitespace → default s100)
SOC_RAW=$(cat /sys/class/boardinfo/soc_name 2>/dev/null | tr 'A-Z' 'a-z' | tr -d '()' | xargs)
SOC="${SOC_RAW:-s100}"

# Map SOC to the corresponding pre-quantized PP-OCRv6 model variant.
# Only S100 and S600 builds are published; everything else (S100P, unknown)
# falls back to the S100 build.
case "$SOC" in
  s600) MODEL_SOC="s600" ;;
  *)    MODEL_SOC="s100" ;;
esac

echo "SOC           : $SOC"
echo "Model variant : rdk_${MODEL_SOC}"

MODEL_BASE_URL="https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_${MODEL_SOC}/paddle_ocr"

# Environment Setup
#
# We only install a Python package when it is NOT importable at all.
# The pinned versions below are the *known-good fallback* used when the system
# has nothing installed; newer pre-installed versions (e.g. Pillow >= 10 or
# newer pyclipper on the S600 image) are accepted as-is.
PYTHON_BIN=python3
PIP_BIN=pip3

REQUIREMENTS=(
  "numpy:numpy==1.26.4"
  "cv2:opencv-python==4.11.0.86"
  "pyclipper:pyclipper==1.3.0.post6"
  "PIL:Pillow==9.0.1"
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
    $PIP_BIN install "${pkg_spec}"
  fi
}

for entry in "${REQUIREMENTS[@]}"; do
  check_and_install "$entry"
done

# Model Download (PP-OCRv6, SOC-aware)
DET_MODEL_PATH="/opt/hobot/model/${MODEL_SOC}/basic/PP-OCRv6_det_infer-deploy_640x640_nv12.hbm"
DET_MODEL_URL="${MODEL_BASE_URL}/PP-OCRv6_det_infer-deploy_640x640_nv12.hbm"

REC_MODEL_PATH="/opt/hobot/model/${MODEL_SOC}/basic/PP-OCRv6_rec_infer-deploy_48x320_rgb.hbm"
REC_MODEL_URL="${MODEL_BASE_URL}/PP-OCRv6_rec_infer-deploy_48x320_rgb.hbm"

# Use whichever download tool is available
if command -v wget &>/dev/null; then
  DL_CMD="wget -q -O"
elif command -v curl &>/dev/null; then
  DL_CMD="curl -fL -o"
else
  echo "ERROR: neither wget nor curl found" >&2
  exit 1
fi

echo "Det model  : $DET_MODEL_PATH"
if [[ ! -f "$DET_MODEL_PATH" ]]; then
  echo "Detection model not found, downloading..."
  mkdir -p "$(dirname "$DET_MODEL_PATH")"
  $DL_CMD "$DET_MODEL_PATH" "$DET_MODEL_URL"
  echo "Detection model downloaded successfully"
else
  echo "Detection model already exists, skip download"
fi

echo "Rec model  : $REC_MODEL_PATH"
if [[ ! -f "$REC_MODEL_PATH" ]]; then
  echo "Recognition model not found, downloading..."
  mkdir -p "$(dirname "$REC_MODEL_PATH")"
  $DL_CMD "$REC_MODEL_PATH" "$REC_MODEL_URL"
  echo "Recognition model downloaded successfully"
else
  echo "Recognition model already exists, skip download"
fi

# Model Execution
python3 main.py \
    --det-model-path "$DET_MODEL_PATH" \
    --rec-model-path "$REC_MODEL_PATH" \
    --test-img ../../test_data/gt_2322.jpg \
    --label-file ../../test_data/ppocrv6_dict.txt \
    --img-save-path result.jpg
