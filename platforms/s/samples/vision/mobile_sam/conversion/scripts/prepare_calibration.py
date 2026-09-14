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

"""Prepare normalized MobileSAM encoder calibration tensors."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32).reshape(3, 1, 1)
STD = np.array([58.395, 57.12, 57.375], dtype=np.float32).reshape(3, 1, 1)


def iter_images(src: Path):
    """Yield image files under a calibration source directory.

    Args:
        src: Directory to search recursively.

    Yields:
        Paths whose suffix matches a supported image extension.
    """

    suffixes = {".jpg", ".jpeg", ".png", ".bmp"}
    for path in sorted(src.rglob("*")):
        if path.suffix.lower() in suffixes:
            yield path


def main() -> None:
    """Create normalized NCHW float32 raw tensors from calibration images."""

    parser = argparse.ArgumentParser(description="Prepare normalized MobileSAM encoder calibration data.")
    parser.add_argument("--src", required=True, help="Directory containing calibration images")
    parser.add_argument("--out", default="./calibration_data_norm_512", help="hb_compile calibration root directory")
    parser.add_argument("--num", type=int, default=30, help="Maximum number of images")
    parser.add_argument("--size", type=int, default=512, help="Square input size")
    args = parser.parse_args()

    src = Path(args.src)
    out = Path(args.out)
    out = out / "normalized_images"
    out.mkdir(parents=True, exist_ok=True)

    count = 0
    for image_path in iter_images(src):
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (args.size, args.size), interpolation=cv2.INTER_LINEAR)
        chw = image.transpose(2, 0, 1).astype(np.float32)
        normalized = (chw - MEAN) / STD
        np.save(out / f"cal_{count:03d}.npy", normalized[None, ...].astype(np.float32))
        count += 1
        if count >= args.num:
            break

    if count == 0:
        raise SystemExit(f"No calibration images found in {src}")
    print(f"Wrote {count} calibration tensors to {out}")


if __name__ == "__main__":
    main()
