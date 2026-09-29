#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
: "${BOARD:?Set BOARD=s100 or s100p explicitly}"
case "$BOARD" in s100|s100p) ;; *) echo 'BOARD must be s100 or s100p' >&2; exit 2;; esac
exec python3 "$HERE/../launcher.py" --target "$BOARD" "$@"
