# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Prepare a cropped plate for the independently converted Sirius S models."""
from pathlib import Path
import argparse
import numpy as np


def prepare_bgr(image: np.ndarray) -> np.ndarray:
    import cv2
    if image is None or image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("Expected a decoded uint8 BGR plate image.")
    image = cv2.resize(image, (94, 24), interpolation=cv2.INTER_LINEAR)
    return np.ascontiguousarray(((image.astype(np.float32) - 127.5) * 0.0078125).transpose(2, 0, 1)[None])


def main(argv=None):
    import cv2
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--image', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(argv)
    tensor = prepare_bgr(cv2.imread(str(a.image)))
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open('xb') as f:
        tensor.tofile(f)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
