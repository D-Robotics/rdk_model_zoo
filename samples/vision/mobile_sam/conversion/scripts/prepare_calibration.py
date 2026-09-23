# Copyright (c) 2025 D-Robotics Corporation
# Licensed under the Apache License, Version 2.0.

"""Prepare MobileSAM ImageNet-normalized encoder calibration tensors."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32).reshape(3, 1, 1)
STD = np.array([58.395, 57.12, 57.375], dtype=np.float32).reshape(3, 1, 1)
TARGETS = ("x5", "s100", "s100p", "s600")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare MobileSAM encoder calibration tensors.")
    parser.add_argument("--target", choices=TARGETS, default="x5")
    parser.add_argument("--src", required=True)
    parser.add_argument("--out", default="./calibration_data_norm_512")
    parser.add_argument("--num", type=int, default=30)
    parser.add_argument("--size", type=int, default=512)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    output = Path(args.out) / ("calibration_data_norm_512" if args.target == "x5" else "normalized_images")
    output.mkdir(parents=True, exist_ok=True)
    suffix = ".rgbchw" if args.target == "x5" else ".npy"
    for old in output.glob(f"*{suffix}"):
        old.unlink()
    images = [p for p in Path(args.src).rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
    if not images:
        raise RuntimeError(f"No calibration images found in {args.src}")
    count = 0
    for path in images[: args.num]:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        chw = cv2.resize(rgb, (args.size, args.size), interpolation=cv2.INTER_LINEAR).transpose(2, 0, 1).astype(np.float32)
        tensor = ((chw - MEAN) / STD)[None].astype(np.float32)
        if args.target == "x5":
            tensor.tofile(output / f"cal_{count:03d}.rgbchw")
        else:
            np.save(output / f"cal_{count:03d}.npy", tensor)
        count += 1
    if count == 0:
        raise RuntimeError(f"No calibration images found in {args.src}")
    print(f"Wrote {count} calibration tensors to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
