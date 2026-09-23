# Copyright (c) 2025 D-Robotics Corporation
# Licensed under the Apache License, Version 2.0.

"""Prepare EfficientSAM encoder calibration tensors for one target family."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

TARGETS = ("x5", "s100", "s100p", "s600")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare EfficientSAM encoder calibration tensors.")
    parser.add_argument("--target", choices=TARGETS, default="x5")
    parser.add_argument("--src", "--image-dir", dest="image_dir", required=True)
    parser.add_argument("--out", "--output-dir", dest="output_dir", required=True)
    parser.add_argument("--num", type=int, default=30)
    parser.add_argument("--size", "--image-size", dest="image_size", type=int, default=512)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = Path(args.output_dir) / ("calibration_data_rgbchw_512" if args.target == "x5" else "batched_images")
    output_dir.mkdir(parents=True, exist_ok=True)
    for old_file in output_dir.glob("*.rgbchw" if args.target == "x5" else "*.npy"):
        old_file.unlink()
    images = [p for p in Path(args.image_dir).rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
    if not images:
        raise RuntimeError(f"No calibration images found in {args.image_dir}")
    while len(images) < min(args.num, 20):
        images.extend(images)
    count = 0
    for image_path in images[: args.num]:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (args.image_size, args.image_size), interpolation=cv2.INTER_LINEAR)
        tensor = np.transpose(image, (2, 0, 1))[None].astype(np.float32) / 255.0
        if args.target == "x5":
            tensor.tofile(output_dir / f"cal_{count:03d}.rgbchw")
        else:
            np.save(output_dir / f"cal_{count:03d}.npy", tensor)
        count += 1
    if count < 20:
        raise RuntimeError(f"Need at least 20 calibration files, got {count}")
    print(f"Wrote {count} calibration files to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
