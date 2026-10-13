#!/usr/bin/env bash
# Run the built replay binary, forwarding arguments and the exit code. The
# tracker is a CPU-only library identical on every supported board, so no
# board detection is needed here.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

bin="$SCRIPT_DIR/build/hobot_mot_replay"
if [ ! -x "$bin" ]; then
  echo "Error: binary not built at $bin" >&2
  echo "Build with: cmake -B $SCRIPT_DIR/build -S $SCRIPT_DIR && cmake --build $SCRIPT_DIR/build" >&2
  exit 2
fi
exec "$bin" "$@"
