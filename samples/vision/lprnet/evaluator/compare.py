# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0

"""Compare saved legacy and unified LPRNet evidence without a board SDK."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from samples.vision.lprnet.runtime.python.lprnet import decode_plate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare LPRNet legacy/unified raw evidence.")
    parser.add_argument("--legacy-raw", type=Path, required=True, help="Legacy raw float32 file")
    parser.add_argument("--unified-raw", type=Path, required=True, help="Unified raw float32 file")
    parser.add_argument("--shape", nargs=3, type=int, default=[1, 68, 18])
    parser.add_argument("--output", type=Path, help="Optional JSON report path")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    shape = tuple(args.shape)
    legacy = np.fromfile(args.legacy_raw, dtype=np.float32)
    unified = np.fromfile(args.unified_raw, dtype=np.float32)
    if legacy.size != int(np.prod(shape)) or unified.size != int(np.prod(shape)):
        raise ValueError(f"both files must contain exactly {int(np.prod(shape))} float32 values")
    legacy = legacy.reshape(shape)
    unified = unified.reshape(shape)
    equal = bool(np.array_equal(legacy, unified))
    report = {
        "shape": list(shape), "dtype": "float32", "array_equal": equal,
        "legacy_plate": decode_plate(legacy[0]), "unified_plate": decode_plate(unified[0]),
        "plate_equal": decode_plate(legacy[0]) == decode_plate(unified[0]),
        "status": "passed" if equal else "failed",
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0 if equal else 2


if __name__ == "__main__":
    raise SystemExit(main())
