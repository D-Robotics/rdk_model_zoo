#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SAMPLE_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

CHIP="${CHIP:-${RDK_SOC:-}}"
if [ -z "${CHIP}" ] && [ -r /sys/class/boardinfo/soc_name ]; then
  CHIP="$(tr '[:upper:]' '[:lower:]' </sys/class/boardinfo/soc_name)"
fi
case "${CHIP}" in
  s100p)
    MODEL_NAME="diffusiondrive_r34_256x1024_s100p.hbm"
    ;;
  s600)
    MODEL_NAME="diffusiondrive_r34_256x1024_s600.hbm"
    ;;
  *)
    echo "Unsupported or undetected RDK platform: ${CHIP:-unknown}" >&2
    exit 1
    ;;
esac
MODEL_PATH="${MODEL_PATH:-${SAMPLE_DIR}/model/${CHIP}/${MODEL_NAME}}"

if [ ! -f "${MODEL_PATH}" ]; then
  CHIP="${CHIP}" MODEL_PATH="${MODEL_PATH}" bash "${SAMPLE_DIR}/model/download_model.sh"
fi

cd "${SCRIPT_DIR}"
python3 main.py --platform "${CHIP}" --model-path "${MODEL_PATH}" "$@"
