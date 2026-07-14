#!/bin/bash
set -euo pipefail

MODEL_DIR="/opt/hobot/model/s100/basic/paraformer"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${1:-${SCRIPT_DIR}/../../test_data}"
VENV_DIR="${SCRIPT_DIR}/.venv"

if [[ ! -f "${MODEL_DIR}/paraformer_large_encoder_400x560_s100.hbm" || ! -f "${MODEL_DIR}/am.mvn" ]]; then
  bash "${SCRIPT_DIR}/../../model/download_model.sh"
fi

if [[ ! -f "${DATA_DIR}/manifest.json" || ! -d "${DATA_DIR}/audio" ]]; then
  echo "Expected ${DATA_DIR}/manifest.json and ${DATA_DIR}/audio containing 16 kHz WAV files." >&2
  exit 1
fi

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  python3 -m venv --system-site-packages "${VENV_DIR}"
fi

PYTHON="${VENV_DIR}/bin/python"
if ! "${PYTHON}" -c 'import funasr, soundfile, torch, torchaudio; assert torch.__version__.split("+")[0] == "2.6.0"; assert torchaudio.__version__.split("+")[0] == "2.6.0"' >/dev/null 2>&1; then
  "${PYTHON}" -m pip install --upgrade pip
  "${PYTHON}" -m pip install --index-url https://download.pytorch.org/whl/cpu 'torch==2.6.0' 'torchaudio==2.6.0'
  "${PYTHON}" -m pip install 'numpy<2' 'protobuf<=4.23.0' 'funasr==1.3.14' 'soundfile==0.14.0'
fi

"${PYTHON}" "${SCRIPT_DIR}/main.py" \
  --manifest "${DATA_DIR}/manifest.json" \
  --audio-dir "${DATA_DIR}/audio" \
  --cmvn-path "${SCRIPT_DIR}/../../model/am.mvn" \
  --max-utts "${N_UTT:-0}"
