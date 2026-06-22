#!/usr/bin/env bash
set -e

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
cd "${SCRIPT_DIR}"

(cd ../../model && bash download_model.sh s100 lite0)

python3 main.py \
  --model-path "../../model/s100/efficientnet_lite0_224x224_nv12.hbm" \
  --test-img "../../test_data/Scottish_deerhound.JPEG" \
  --label-file "../../test_data/imagenet_classes.names" \
  --top-k 5 \
  "$@"
