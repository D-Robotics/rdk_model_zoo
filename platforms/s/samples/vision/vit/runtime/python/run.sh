#!/usr/bin/env bash
set -e

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
cd "${SCRIPT_DIR}"

MODEL_VARIANT=${1:-int8}
shift || true

(cd ../../model && bash download_model.sh s100 "${MODEL_VARIANT}")

python3 main.py \
  --model-variant "${MODEL_VARIANT}" \
  --test-img "../../test_data/airplane_0000.png" \
  --label-file "../../test_data/cifar10_classes.names" \
  --top-k 5 \
  "$@"
