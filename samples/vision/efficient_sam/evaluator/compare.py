# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Capture same-board legacy/unified segmentation evidence; never downloads."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
from samples._shared.sam_evaluator import main

if __name__ == "__main__":
    raise SystemExit(main('efficient_sam'))
