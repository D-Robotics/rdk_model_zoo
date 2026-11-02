#!/usr/bin/env python3
# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Prepare verified COCO images as normalized Small model calibration tensors."""

import argparse
import json
from pathlib import Path

from export_small import digest, preprocess


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    import cv2
    import numpy as np

    manifest_path = args.images / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    rows = manifest["images"]
    if len(rows) != manifest["count"] or len({row["name"] for row in rows}) != len(rows):
        raise ValueError("Calibration source manifest count/uniqueness mismatch")
    # A nested images/ directory binds one input name; metadata stays outside it.
    tensors = args.output_dir / "images"
    tensors.mkdir(parents=True, exist_ok=False)
    result = []
    for row in rows:
        source = args.images / row["name"]
        if digest(source) != row["sha256"]:
            raise ValueError(f"Calibration source digest mismatch: {source}")
        value = preprocess(cv2.imread(str(source)))
        if not np.isfinite(value).all():
            raise ValueError(f"Nonfinite calibration tensor: {source}")
        destination = tensors / (source.stem + ".npy")
        np.save(destination, value)
        result.append({"source": row["name"], "source_sha256": row["sha256"],
                       "tensor": str(destination.relative_to(args.output_dir)),
                       "tensor_sha256": digest(destination)})
    report = {
        "count": len(result), "seed": manifest["seed"], "dataset": manifest["dataset"],
        "source_manifest_sha256": digest(manifest_path), "producer_sha256": digest(__file__),
        "preprocess_source_sha256": digest(Path(__file__).with_name("export_small.py")),
        "input_name": "images", "shape": [1, 3, 518, 686], "dtype": "float32",
        "preprocess": "RGB float64/255; cubic stretch 518x686; ImageNet mean/std; F32 NCHW",
        "license_note": manifest.get("license_note"), "tensors": result,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"count": len(result), "manifest_sha256": digest(args.output_dir / "manifest.json")}))


if __name__ == "__main__":
    main()
