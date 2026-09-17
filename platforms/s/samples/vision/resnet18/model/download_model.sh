#!/usr/bin/env bash
set -euo pipefail

SOC="${1:-s100}"
case "$SOC" in
  s100|s600) ;;
  *)
    echo "Unsupported SoC: $SOC" >&2
    echo "Available: s100, s600" >&2
    exit 1
    ;;
esac

# The canonical downloader owns the manifest lookup and atomic download. This
# compatibility path keeps the historical S100/S600 argument and directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CANONICAL="$(cd "$SCRIPT_DIR/../../../../../../samples/vision/resnet/model" && pwd)"
exec python3 "$CANONICAL/download.py" --target "$SOC" --output-dir "$SCRIPT_DIR"
