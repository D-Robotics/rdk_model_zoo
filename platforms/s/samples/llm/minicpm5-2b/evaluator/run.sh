#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
EVAL_BUNDLE=${EVAL_BUNDLE:?Set EVAL_BUNDLE to the copied host-prepared ppl-bundle directory}
MODEL_DIR=${MODEL_DIR:-"$HERE/../model/s600"}
OUTPUT=${OUTPUT:-"$HERE/board-local-ppl.json"}
MODEL_DIR=$(realpath "$MODEL_DIR")
EVAL_BUNDLE=$(realpath "$EVAL_BUNDLE")
export PYTHONPATH="$EVAL_BUNDLE:$HERE/.deps${PYTHONPATH:+:$PYTHONPATH}"
python3 -c 'import grpc, numpy, google.protobuf' || {
  echo "Install board dependencies: python3 -m pip install --target '$HERE/.deps' grpcio==1.74.0 protobuf==4.25.8 numpy" >&2
  exit 1
}
export OPENBLAS_NUM_THREADS=4
export LD_LIBRARY_PATH="$EVAL_BUNDLE/sdk-runtime/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export HB_DNN_USER_DEFINED_L2M_SIZES=6:6:6:6
WORK=$(mktemp -d /tmp/minicpm5-ppl.XXXXXX)
mkdir -p "$WORK/script" "$WORK/log"
SERVER_PID=''
cleanup() {
  if [[ -n "$SERVER_PID" ]]; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  rm -rf -- "$WORK"
}
trap cleanup EXIT
"$EVAL_BUNDLE/sdk-runtime/bin/hbm_rpc_service" --timeout=600 --work_dir="$WORK" \
  --log_level=2 --core_id=0,1,2,3 --custom_port=0 >"${OUTPUT}.service.log" 2>&1 &
SERVER_PID=$!
for ((i=0; i<50; i++)); do
  [[ -s "$WORK/script/available_port.txt" ]] && break
  kill -0 "$SERVER_PID" 2>/dev/null || { cat "${OUTPUT}.service.log"; exit 1; }
  sleep 0.1
done
PORT=$(awk '{print $NF}' "$WORK/script/available_port.txt")
[[ "$PORT" =~ ^[0-9]+$ ]] || { echo "Invalid SDK RPC port" >&2; exit 1; }
python3 -u "$HERE/main.py" --port="$PORT" --data-dir="$EVAL_BUNDLE" \
  --hbm="$MODEL_DIR/MiniCPM5-2B_language_chunk_256_cache_4096_w8_nash-p_corenum_4_4.hbm" \
  --embedding="$MODEL_DIR/MiniCPM5-2B_embed_tokens.bin" --output="$OUTPUT" "$@"
