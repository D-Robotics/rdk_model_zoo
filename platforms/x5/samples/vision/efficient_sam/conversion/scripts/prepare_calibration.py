#!/usr/bin/env python3
"""Prepare EfficientSAM encoder calibration tensors."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def main() -> None:
    """Create RGB CHW float32 raw tensors from calibration images."""

    parser = argparse.ArgumentParser(description="Prepare EfficientSAM encoder calibration tensors.")
    parser.add_argument("--src", "--image-dir", dest="image_dir", required=True, help="Directory containing calibration images.")
    parser.add_argument("--out", "--output-dir", dest="output_dir", required=True, help="Output calibration directory.")
    parser.add_argument("--num", type=int, default=30)
    parser.add_argument("--size", "--image-size", dest="image_size", type=int, default=512)
    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for old_file in output_dir.glob("*.rgbchw"):
        old_file.unlink()

    images = [path for path in image_dir.rglob("*") if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
    if not images:
        raise RuntimeError(f"No calibration images found in {image_dir}")
    while len(images) < min(args.num, 20):
        images.extend(images)
    images = images[: args.num]

    count = 0
    for image_path in images:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (args.image_size, args.image_size), interpolation=cv2.INTER_LINEAR)
        tensor = np.transpose(image, (2, 0, 1))[None].astype(np.float32) / 255.0
        tensor.tofile(output_dir / f"cal_{count:03d}.rgbchw")
        count += 1
    if count < 20:
        raise RuntimeError(f"Need at least 20 calibration files, got {count}")
    print(f"Wrote {count} calibration files to {output_dir}")


if __name__ == "__main__":
    main()