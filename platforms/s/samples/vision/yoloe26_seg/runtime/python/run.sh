#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
SIZE=n
CUSTOM=false
ARGS=("$@")
while (($#)); do
  case "$1" in
    --size) SIZE=${2:?Missing size}; shift 2;;
    --size=*) SIZE=${1#*=}; shift;;
    --model-path|--model-path=*) CUSTOM=true; shift;;
    -h|--help) exec python3 "$SCRIPT_DIR/main.py" --help;;
    *) shift;;
  esac
done
if [[ "$CUSTOM" == false ]]; then
  bash "$SCRIPT_DIR/../../model/download_model.sh" auto "$SIZE"
fi
exec python3 "$SCRIPT_DIR/main.py" "${ARGS[@]}"
