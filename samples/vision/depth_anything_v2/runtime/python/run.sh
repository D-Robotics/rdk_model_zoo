#!/bin/bash
set -e

SOC=$(tr 'A-Z' 'a-z' </sys/class/boardinfo/soc_name)
MODEL_SOC="s100"
echo "SOC        : ${SOC}"
echo "Model SOC  : ${MODEL_SOC}"

# Environment Setup
PYTHON_BIN=python3
PIP_BIN=pip3

REQUIREMENTS=(
  "numpy==1.26.4"
  "opencv-python==4.11.0.86"
  "torch==2.3.1"
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

MODEL_PATH="../../model/${MODEL_SOC}/depth_any.hbm"
(cd ../../model && bash download_model.sh "${MODEL_SOC}")

# Model Execution
python3 main.py \
    --model-path "${MODEL_PATH}" \
    --test-img ../../test_data/furseal.jpg \
    --img-save-path result.jpg
