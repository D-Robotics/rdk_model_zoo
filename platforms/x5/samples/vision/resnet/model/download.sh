#!/usr/bin/env bash
set -euo pipefail

# Keep the historical command path, but let the canonical downloader read the
# published URL/format/hash from platforms/x5/docs/release/models.yaml.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CANONICAL="$(cd "$SCRIPT_DIR/../../../../../../samples/vision/resnet/model" && pwd)"
exec python3 "$CANONICAL/download.py" --target x5 --output-dir "$SCRIPT_DIR"
