#!/bin/bash
# Run the HGNetV2 b0 image classification demo with the default model.
#
# The script prefers a board-side preinstalled model, falls back to the
# repository ``model/`` directory, and finally downloads the prebuilt
# ``.bin`` from the D-Robotics archive if neither exists.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="${SCRIPT_DIR}/../../model"
MODEL_NAME="hgnetv2_b0_224x224_nv12.bin"

MODEL_PATH="/opt/hobot/model/x5/basic/${MODEL_NAME}"
[ ! -f "${MODEL_PATH}" ] && MODEL_PATH="${MODEL_DIR}/${MODEL_NAME}"
if [ ! -f "${MODEL_PATH}" ]; then
    bash "${MODEL_DIR}/download.sh"
    MODEL_PATH="${MODEL_DIR}/${MODEL_NAME}"
fi

cd "${SCRIPT_DIR}"
python3 main.py --model-path "${MODEL_PATH}" "$@"
