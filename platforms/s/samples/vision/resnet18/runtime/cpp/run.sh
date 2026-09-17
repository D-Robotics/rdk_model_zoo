#!/usr/bin/env bash
set -euo pipefail

# Preserve the historical launcher path while delegating build/run behavior to
# the canonical S18 source. No apt, pip, or model download is performed here.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CANONICAL="$(cd "$SCRIPT_DIR/../../../../../../../samples/vision/resnet/runtime/cpp" && pwd)"
OLD_SAMPLE="$(cd "$SCRIPT_DIR/../.." && pwd)"
PLATFORM_ROOT="$(cd "$SCRIPT_DIR/../../../../../" && pwd)"

MODEL_PATH="${MODEL_PATH:-$OLD_SAMPLE/model/s100/resnet18_224x224_nv12.hbm}"
TEST_IMAGE="${TEST_IMAGE:-$OLD_SAMPLE/test_data/zebra_cls.jpg}"
LABEL_FILE="${LABEL_FILE:-$PLATFORM_ROOT/datasets/imagenet/imagenet_classes.names}"

MODEL_PATH="$MODEL_PATH" \
TEST_IMAGE="$TEST_IMAGE" \
LABEL_FILE="$LABEL_FILE" \
exec bash "$CANONICAL/run.sh" "$@"
