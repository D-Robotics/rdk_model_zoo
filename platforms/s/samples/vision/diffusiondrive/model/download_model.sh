#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARCHIVE_BASE_URL="${ARCHIVE_BASE_URL:-https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100}"
CHIP="${CHIP:-${RDK_SOC:-${1:-}}}"
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
    echo "Usage: CHIP=s100p|s600 bash download_model.sh" >&2
    exit 1
    ;;
esac

RELATIVE_MODEL_PATH="${CHIP}/${MODEL_NAME}"
MODEL_PATH="${MODEL_PATH:-${SCRIPT_DIR}/${RELATIVE_MODEL_PATH}}"
MODEL_URL="${MODEL_URL:-${ARCHIVE_BASE_URL}/${MODEL_NAME}}"

verify_model() {
  local candidate expected actual
  candidate="$1"
  expected="$(awk -v path="${RELATIVE_MODEL_PATH}" '$2 == path {print $1}' "${SCRIPT_DIR}/SHA256SUMS")"
  if [ -z "${expected}" ]; then
    echo "No checksum entry for ${RELATIVE_MODEL_PATH}" >&2
    return 1
  fi
  actual="$(sha256sum "${candidate}" | awk '{print $1}')"
  if [ "${actual}" != "${expected}" ]; then
    echo "Checksum mismatch for ${candidate}" >&2
    return 1
  fi
  echo "Checksum OK: ${candidate}"
}

if [ -f "${MODEL_PATH}" ]; then
  echo "Model already exists: ${MODEL_PATH}"
  verify_model "${MODEL_PATH}"
  exit $?
fi

mkdir -p "$(dirname "${MODEL_PATH}")"
wget -c "${MODEL_URL}" -O "${MODEL_PATH}.part"
verify_model "${MODEL_PATH}.part"
mv "${MODEL_PATH}.part" "${MODEL_PATH}"
