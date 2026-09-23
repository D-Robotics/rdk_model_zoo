#!/usr/bin/env bash
# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 s100 [output-dir]" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ $# -gt 1 ]]; then
  python3 "${SCRIPT_DIR}/download.py" --target "$1" --output-dir "$2"
else
  python3 "${SCRIPT_DIR}/download.py" --target "$1"
fi
