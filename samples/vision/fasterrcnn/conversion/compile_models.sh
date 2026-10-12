#!/usr/bin/env bash
# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Compile the paired artifacts for the running container family.
#
# Usage (inside the compile container, from the conversion directory):
#   ./compile_models.sh x5     # X5 container: hb_mapper makertbin (bayes-e)
#   ./compile_models.sh s      # S container: hb_compile (s100/s100p/s600)
#
# Each compile writes its log next to the config (build/<target>/<part>/
# compile.log) and the script stops at the first failure.  The compiler
# prints the model input/output interface table (names, shapes, dtypes,
# quant scales) into that log — verify_pair.py summarizes the artifacts.
set -euo pipefail

FAM="${1:?usage: compile_models.sh x5|s}"
HERE="$(cd "$(dirname "$0")" && pwd)"
BUILD="${HERE}/build"

compile_one() { # config_path, command...
  local config="$1"
  shift
  local log="${config%.yaml}.compile.log"
  echo "=== ${config} ==="
  if ! "$@" | tee "${log}" >&2; then
    echo "compile failed; see ${log}" >&2
    exit 1
  fi
}

if [ "${FAM}" = "x5" ]; then
  compile_one "${BUILD}/x5/parta/config.yaml" hb_mapper makertbin \
    --config "${BUILD}/x5/parta/config.yaml" --model-type onnx
  compile_one "${BUILD}/x5/partb/config.yaml" hb_mapper makertbin \
    --config "${BUILD}/x5/partb/config.yaml" --model-type onnx
elif [ "${FAM}" = "s" ]; then
  for target in s100 s100p s600; do
    march="$(grep -m1 '^  march:' "${BUILD}/${target}/parta/config.yaml" | awk '{print $2}')"
    compile_one "${BUILD}/${target}/parta/config.yaml" hb_compile \
      --march "${march}" -c "${BUILD}/${target}/parta/config.yaml"
    compile_one "${BUILD}/${target}/partb/config.yaml" hb_compile \
      --march "${march}" -c "${BUILD}/${target}/partb/config.yaml"
  done
else
  echo "unknown family: ${FAM} (expected x5 or s)" >&2
  exit 2
fi
echo "ALL DONE"
