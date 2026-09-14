#!/bin/bash
set -e

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
cd "${SCRIPT_DIR}"

SOC=$(tr 'A-Z' 'a-z' </sys/class/boardinfo/soc_name)
MODEL_SOC="${MODEL_SOC:-${SOC}}"
echo "SOC        : ${SOC}"
echo "Model SOC  : ${MODEL_SOC}"

# Add 3rdparty tracker to Python path
export PYTHONPATH="${SCRIPT_DIR}/../../3rdparty:${PYTHONPATH}"

# Environment Setup
PIP_BIN=pip3

REQUIREMENTS=(
  "numpy==1.26.4"
  "opencv-python==4.11.0.86"
  "scipy==1.15.3"
  "lap==0.5.12"
  "cython-bbox==0.1.5"
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

MODEL_PATH="../../model/${MODEL_SOC}/yolov5x_672x672_nv12.hbm"
(cd ../../model && bash download_model.sh "${MODEL_SOC}")

TEST_VIDEO="../../test_data/track_test.mp4"
if [[ ! -f "${TEST_VIDEO}" ]]; then
  echo "Downloading ${TEST_VIDEO}..."
  wget -O "${TEST_VIDEO}" "https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/ByteTrack/track_test.mp4"
fi

# Model Execution
python3 main.py \
    --model-path "${MODEL_PATH}" \
    --input "${TEST_VIDEO}" \
    --output result.mp4 \
    --score-thres 0.25 \
    --track-thresh 0.3
