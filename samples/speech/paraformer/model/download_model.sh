#!/bin/bash
set -euo pipefail

SOC=$(tr 'A-Z' 'a-z' </sys/class/boardinfo/soc_name)
if [[ "${SOC}" != "s100" ]]; then
  echo "Paraformer is currently released only for RDK S100; detected ${SOC}." >&2
  exit 1
fi

MODEL_DIR="/opt/hobot/model/s100/basic/paraformer"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_URL="https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/paraformer/nash-e"
declare -A FILES=(
  [encoder_int16.hbm]=paraformer_large_encoder_400x560_s100.hbm
  [predictor_int16.hbm]=paraformer_large_predictor_400x512_s100.hbm
  [decoder_int16.hbm]=paraformer_large_decoder_400x512_s100.hbm
  [tokens.json]=tokens.json
)
mkdir -p "${MODEL_DIR}"

for remote_file in "${!FILES[@]}"; do
  local_file="${FILES[${remote_file}]}"
  if [[ -f "${MODEL_DIR}/${local_file}" ]]; then
    echo "Exists: ${MODEL_DIR}/${local_file}"
    continue
  fi
  echo "Downloading ${remote_file}..."
  if command -v curl >/dev/null 2>&1; then
    curl -fL "${BASE_URL}/${remote_file}" -o "${MODEL_DIR}/${local_file}"
  elif command -v wget >/dev/null 2>&1; then
    wget -q --show-progress -O "${MODEL_DIR}/${local_file}" "${BASE_URL}/${remote_file}"
  else
    echo "Neither curl nor wget is available; install one download client and retry." >&2
    exit 1
  fi
done

for frontend_file in am.mvn paraformer_config.yaml; do
  if [[ ! -f "${MODEL_DIR}/${frontend_file}" ]]; then
    cp "${SCRIPT_DIR}/${frontend_file}" "${MODEL_DIR}/${frontend_file}"
  fi
done

echo "Paraformer model package is ready: ${MODEL_DIR}"
