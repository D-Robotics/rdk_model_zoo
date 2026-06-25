#!/usr/bin/env python3
"""Generate synthetic calibration images (gradients, noise, patterns).

Note: Real COCO images are recommended for best quantization accuracy.
This script is provided as a fallback when network access is unavailable.

Usage:
    python generate_calib_images.py [--output-dir ../../calibration_data/images]
"""
import argparse
import os
from pathlib import Path

import numpy as np
from PIL import Image

DEFAULT_OUT = Path(__file__).resolve().parent.parent.parent / "calibration_data" / "images"


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate synthetic calibration images")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(42)

    sizes = [(224, 224), (336, 336), (560, 560), (896, 896), (1024, 1024)]
    count = 0

    for i, (height, width) in enumerate(sizes):
        gradient = np.zeros((height, width, 3), dtype=np.uint8)
        for y in range(height):
            gradient[y, :, 0] = int(255 * y / height)
            gradient[y, :, 1] = int(255 * (1 - y / height))
            gradient[y, :, 2] = 128
        Image.fromarray(gradient, "RGB").save(output_dir / f"gradient_{i}_{height}x{width}.jpg", "JPEG")
        count += 1

        noise = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        Image.fromarray(noise, "RGB").save(output_dir / f"noise_{i}_{height}x{width}.jpg", "JPEG")
        count += 1

        pattern = np.zeros((height, width, 3), dtype=np.uint8)
        for y in range(height):
            for x in range(width):
                if (x // 32 + y // 32) % 2 == 0:
                    pattern[y, x] = [255, 255, 255]
        Image.fromarray(pattern, "RGB").save(output_dir / f"pattern_{i}_{height}x{width}.jpg", "JPEG")
        count += 1

        stripes = np.zeros((height, width, 3), dtype=np.uint8)
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
        sw = width // len(colors)
        for j, color in enumerate(colors):
            start = j * sw
            end = (j + 1) * sw if j < len(colors) - 1 else width
            stripes[:, start:end] = color
        Image.fromarray(stripes, "RGB").save(output_dir / f"stripes_{i}_{height}x{width}.jpg", "JPEG")
        count += 1

    print(f"Generated {count} synthetic images -> {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
