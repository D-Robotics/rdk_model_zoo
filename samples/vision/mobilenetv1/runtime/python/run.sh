#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="${SCRIPT_DIR}/../../model"
MODEL_PATH="../../model/s100/mobilenetv1_224x224_nv12.hbm"

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
    $PIP_BIN install "$name==$version" --break-system-packages
  fi
}

for pkg in "${REQUIREMENTS[@]}"; do
  check_and_install "$pkg"
done

echo "Model path : $MODEL_PATH"

if [[ ! -f "$MODEL_PATH" ]]; then
  echo "Model not found, downloading..."
  (cd "$MODEL_DIR" && bash download_model.sh s100)
else
  echo "Model already exists, skip download"
fi

cd "$SCRIPT_DIR"
"$PYTHON_BIN" main.py \
  --model-path "$MODEL_PATH" \
  --test-img ../../test_data/zebra_cls.jpg \
  --label-file ../../test_data/imagenet_classes.names \
  --top-k 5 \
  "$@"
