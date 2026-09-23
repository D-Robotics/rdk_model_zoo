# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0

"""Compare saved legacy and unified MODNet matte evidence offline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare MODNet legacy/unified matte evidence.")
    parser.add_argument("--legacy-matte", type=Path, required=True)
    parser.add_argument("--unified-matte", type=Path, required=True)
    parser.add_argument("--atol", type=float, default=0.0, help="Allowed uint8 absolute difference")
    parser.add_argument("--output", type=Path)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.atol < 0:
        raise ValueError("--atol must be non-negative")
    legacy = cv2.imread(str(args.legacy_matte), cv2.IMREAD_GRAYSCALE)
    unified = cv2.imread(str(args.unified_matte), cv2.IMREAD_GRAYSCALE)
    if legacy is None or unified is None:
        raise FileNotFoundError("both matte files must be readable grayscale images")
    if legacy.shape != unified.shape:
        raise ValueError(f"matte shapes differ: {legacy.shape} vs {unified.shape}")
    diff = np.abs(legacy.astype(np.int16) - unified.astype(np.int16))
    max_abs = int(diff.max(initial=0))
    report = {
        "shape": list(legacy.shape), "dtype": "uint8", "max_abs": max_abs,
        "mean_abs": float(diff.mean()), "atol": args.atol,
        "array_equal": bool(np.array_equal(legacy, unified)),
        "status": "passed" if max_abs <= args.atol else "failed",
    }
    print(json.dumps(report, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0 if max_abs <= args.atol else 2


if __name__ == "__main__":
    raise SystemExit(main())
