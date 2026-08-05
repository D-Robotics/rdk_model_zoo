#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SAMPLE_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
RESULT_DIR="${1:-${SCRIPT_DIR}/results}"

mkdir -p "${RESULT_DIR}"
for case_dir in "${SAMPLE_DIR}"/test_data/case_*; do
  case_name="$(basename "${case_dir}")"
  case_result_dir="${RESULT_DIR}/${case_name}"
  mkdir -p "${case_result_dir}"
  echo "[Run] ${case_name}"
  bash "${SCRIPT_DIR}/run.sh" \
    --input-npz "${case_dir}/inputs.npz" \
    --output-npz "${case_result_dir}/outputs.npz" \
    --output-image "${case_result_dir}/result.png"
done

echo "[Saved] Multi-case results: ${RESULT_DIR}"
