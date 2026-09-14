#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
BOARD=${BOARD:-s100}
case "$BOARD" in s100|s100p) ;; *) echo 'BOARD must be s100 or s100p' >&2; exit 2;; esac
RUNTIME_ROOT=${OELLM_RUNTIME_ROOT:-${OELLM_SDK_ROOT:+$OELLM_SDK_ROOT/oellm_runtime}}
: "${RUNTIME_ROOT:?Set OELLM_RUNTIME_ROOT or OELLM_SDK_ROOT to SDK 1.0.0}"
: "${EVAL_BUNDLE:?Set EVAL_BUNDLE to the prepared input directory}"
export BOARD MODEL_DIR=${MODEL_DIR:-"$HERE/../../model/$BOARD"}
bash "$HERE/../../model/download_model.sh"
export LD_LIBRARY_PATH="$RUNTIME_ROOT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
python3 "$HERE/evaluate.py" --model "$MODEL_DIR/minicpm5-2b_ctx4096_${BOARD}.hbm" \
  --input-ids "$EVAL_BUNDLE/input_ids.npy" --output "${OUTPUT:-legacy-ppl.json}" "$@"
