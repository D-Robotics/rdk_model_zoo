#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
sample_dir="$(cd -- "$script_dir/../.." && pwd)"
repo_root="$(cd -- "$sample_dir/../../.." && pwd)"
variant="${VARIANT:-spatial-rgb-late8}"
target="${TARGET:-x5}"
model_path="${MODEL_PATH:-}"
fixture_dir="${FIXTURE_DIR:-$sample_dir/test_data/fixture}"
build_dir="${BUILD_DIR:-$script_dir/build}"
model_path="$(PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}" "${PYTHON_BIN:-python3}" - "$model_path" "$variant" "$target" "$sample_dir" <<'PY'
import sys
from utils.py_utils.platforms import require_execution_target
from pathlib import Path
from samples.vision.osnet.model.artifacts import require_artifact, load_manifest
require_execution_target(sys.argv[3])
path = Path(sys.argv[1]) if sys.argv[1] else Path(sys.argv[4]) / "model" / load_manifest()["artifacts"][sys.argv[2]]["filename"]
record = require_artifact(path, sys.argv[2])
if record.get("target", "x5") != sys.argv[3]:
    raise ValueError("Artifact target differs from TARGET")
print(path)
PY
)"
test -f "$fixture_dir/images.dat"
test -f "$fixture_dir/$variant.f32"
mkdir -p "$build_dir"
flags=(-ldnn)
if [[ "$target" != x5 ]]; then
  flags=(-DOSNET_UCP -I/usr/hobot/include -L/usr/hobot/lib -Wl,-rpath,/usr/hobot/lib -ldnn -lhbucp)
  export LD_LIBRARY_PATH="/usr/hobot/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
"${CXX:-g++}" -O3 -std=c++17 -Wall -Wextra "$script_dir/main.cpp" "$script_dir/osnet.cpp" -pthread "${flags[@]}" -o "$build_dir/osnet"
if [[ "${BENCHMARK:-0}" == 1 ]]; then
  exec "$build_dir/osnet" --benchmark "$model_path" "$fixture_dir/images.dat" "$fixture_dir/$variant.f32" \
    "${OUTPUT_JSON:-$build_dir/result.json}" "${STREAMS:-1}" "${WARMUP:-20}" "${RUNS:-200}" "${ROUNDS:-3}"
fi
exec "$build_dir/osnet" "$model_path" "$fixture_dir/images.dat" "$fixture_dir/$variant.f32" \
  "${OUTPUT_JSON:-$build_dir/result.json}" "${RUNS:-3000}"
