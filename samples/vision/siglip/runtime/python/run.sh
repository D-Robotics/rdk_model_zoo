#!/bin/bash

set -e

cd "$(dirname "$0")"

MODEL_PATH=${MODEL_PATH:-../../model/s100/bpu-siglip-base-patch16-224.hbm}
TEST_IMG=${TEST_IMG:-../../test_data/dog.jpg}
IMAGE_SIZE=${IMAGE_SIZE:-224}
SUBMODEL=${1:-pooler_output}

bash ../../model/download_model.sh bpu-siglip-base-patch16-224

python3 main.py \
  --model-path "${MODEL_PATH}" \
  --test-img "${TEST_IMG}" \
  --image-size "${IMAGE_SIZE}" \
  --submodel "${SUBMODEL}"
