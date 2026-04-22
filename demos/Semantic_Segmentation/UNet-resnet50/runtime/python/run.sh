#!/bin/bash
# UNet-resnet50 Python Inference One-Click Run Script
# This script configures the environment, downloads the model if needed,
# and runs the Python inference example.

set -e

# ---------------------------------------------------------------------------
# 0. Environment configuration
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL_DIR="${SCRIPT_DIR}/../../model"
TEST_DATA_DIR="${SCRIPT_DIR}/../../test_data"

echo "[INFO] Script directory: ${SCRIPT_DIR}"

# ---------------------------------------------------------------------------
# 1. Model check
# ---------------------------------------------------------------------------
MODEL_NAME="unet_resnet50_512x512_nv12.bin"
MODEL_PATH="${MODEL_DIR}/${MODEL_NAME}"

if [ ! -f "${MODEL_PATH}" ]; then
    echo "[INFO] Model not found at ${MODEL_PATH}"
    echo "[INFO] Please place the correct .bin model for your platform (RDK X3 or RDK X5)."
    # Try to use system default path
    DEFAULT_MODEL_PATH="/opt/hobot/model/basic/${MODEL_NAME}"
    if [ -f "${DEFAULT_MODEL_PATH}" ]; then
        echo "[INFO] Using system default model: ${DEFAULT_MODEL_PATH}"
        MODEL_PATH="${DEFAULT_MODEL_PATH}"
    else
        echo "[WARN] Default model path not found: ${DEFAULT_MODEL_PATH}"
    fi
else
    echo "[INFO] Using model: ${MODEL_PATH}"
fi

# ---------------------------------------------------------------------------
# 2. Check test data
# ---------------------------------------------------------------------------
TEST_IMAGE="${TEST_DATA_DIR}/UNet_Segmentation_Origin.png"
if [ ! -f "${TEST_IMAGE}" ]; then
    echo "[WARN] Test image not found: ${TEST_IMAGE}"
    echo "[WARN] Please provide a test image."
    TEST_IMAGE=""
fi

# ---------------------------------------------------------------------------
# 3. Run inference
# ---------------------------------------------------------------------------
echo "[INFO] Running UNet-resnet50 inference..."

cd "${SCRIPT_DIR}"

if [ -n "${TEST_IMAGE}" ] && [ -f "${MODEL_PATH}" ]; then
    python3 main.py \
        --model-path "${MODEL_PATH}" \
        --img-path "${TEST_IMAGE}" \
        --save-path "unet_result.jpg" \
        --mask-path "unet_mask.png"
elif [ -f "${MODEL_PATH}" ]; then
    python3 main.py \
        --model-path "${MODEL_PATH}"
else
    python3 main.py
fi

echo "[INFO] Inference completed."
