# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Explicit caller-image calibration, matching canonical runtime input arithmetic."""

import argparse, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
from samples._shared.assets import sha256_file
from samples.vision.lanenet.runtime.python.image_preprocess import image_to_tensor

PROTOCOL = "lanenet-rgb-area-imagenet-f32-nchw-v1"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--images", required=True, type=Path)
    p.add_argument("--count", required=True, type=int)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(args.output)
    if not args.images.is_dir():
        raise ValueError("images must be an existing directory")
    images = sorted(
        path
        for path in args.images.rglob("*")
        if path.is_file() and path.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")
    )
    if args.count <= 0 or args.count > len(images):
        raise ValueError("count must be positive and no greater than available images")
    args.output.mkdir(parents=True)
    data = args.output / "data"
    data.mkdir()
    records = []
    for index, path in enumerate(images[: args.count]):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Cannot decode calibration image: {path}")
        tensor = image_to_tensor(image)
        name = f"data/{index:06d}.npy"
        dest = args.output / name
        np.save(dest, tensor)
        records.append(
            {
                "image": str(path.resolve()),
                "image_sha256": sha256_file(path),
                "image_shape": list(image.shape),
                "tensor": name,
                "tensor_sha256": sha256_file(dest),
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
            }
        )
    report = {
        "schema_version": "1.0",
        "protocol": PROTOCOL,
        "target": "s100",
        "count": len(records),
        "selection": "lexicographic paths; first count; no implicit dataset download",
        "records": records,
        "source_recipe": "new explicit preparation; missing source calibration script is not recovered",
        "oe_validation": "not-run",
    }
    (args.output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Prepared {len(records)} tensors at {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
