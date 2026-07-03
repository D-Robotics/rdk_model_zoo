#!/usr/bin/env python3
import argparse
from pathlib import Path

import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare PP-LiteSeg calibration tensors for hb_mapper.")
    parser.add_argument("--src", type=Path, required=True, help="Directory containing calibration images.")
    parser.add_argument("--out", type=Path, default=Path("calibration_data_rgb_f32_1024x512"), help="Output directory.")
    parser.add_argument("--width", type=int, default=1024, help="Model input width.")
    parser.add_argument("--height", type=int, default=512, help="Model input height.")
    parser.add_argument("--num", type=int, default=50, help="Maximum number of images to export.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed used when sampling images.")
    return parser.parse_args()


def collect_images(src):
    suffixes = {".jpg", ".jpeg", ".png", ".bmp"}
    return sorted(path for path in src.rglob("*") if path.suffix.lower() in suffixes)


def main():
    args = parse_args()
    images = collect_images(args.src)
    if not images:
        raise FileNotFoundError(f"No calibration images found in {args.src}")

    if len(images) > args.num:
        rng = np.random.default_rng(args.seed)
        selected = rng.choice(len(images), args.num, replace=False)
        images = [images[index] for index in sorted(selected)]

    args.out.mkdir(parents=True, exist_ok=True)

    for image_path in images:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            print(f"Skip unreadable image: {image_path}")
            continue

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (args.width, args.height), interpolation=cv2.INTER_LINEAR)
        tensor = np.transpose(image, (2, 0, 1))[None].astype(np.float32)
        tensor.tofile(args.out / f"{image_path.stem}.rgbchw")

    expected_bytes = 1 * 3 * args.height * args.width * 4
    print(f"Wrote {len(list(args.out.glob('*.rgbchw')))} calibration tensors to {args.out}")
    print(f"Expected file size: {expected_bytes} bytes per tensor")


if __name__ == "__main__":
    main()
