#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="${SCRIPT_DIR}/../../model"

IMAGE_MODEL_PATH="${MODEL_DIR}/img_encoder.bin"
TEXT_MODEL_PATH="${MODEL_DIR}/text_encoder.onnx"

# Download models if missing
if [ ! -f "${IMAGE_MODEL_PATH}" ] || [ ! -f "${TEXT_MODEL_PATH}" ]; then
    bash "${MODEL_DIR}/download.sh"
fi

python3 - <<'PY'
import importlib.util
missing = [name for name in ("onnxruntime", "ftfy", "regex") if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit(
        "Missing Python dependencies: "
        + ", ".join(missing)
        + "\nInstall them with: python3 -m pip install --user onnxruntime ftfy regex"
    )
PY

cd "${SCRIPT_DIR}"
python3 main.py "$@"
