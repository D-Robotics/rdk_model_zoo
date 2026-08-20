#!/usr/bin/env python3
# Copyright (c) 2025 D-Robotics Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
    parser.add_argument("--num", type=int, default=30, help="Maximum number of calibration tensors.")
    parser.add_argument("--size", "--image-size", dest="image_size", type=int, default=512, help="Square image size.")
    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    output_dir = Path(args.output_dir)
    output_dir = output_dir / "batched_images"
    output_dir.mkdir(parents=True, exist_ok=True)
    for old_file in output_dir.glob("*.npy"):
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
        np.save(output_dir / f"cal_{count:03d}.npy", tensor)
        count += 1
    if count < 20:
        raise RuntimeError(f"Need at least 20 calibration files, got {count}")
    print(f"Wrote {count} calibration files to {output_dir}")


if __name__ == "__main__":
    main()
