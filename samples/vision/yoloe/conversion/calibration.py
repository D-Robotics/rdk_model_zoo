# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Deterministic calibration files sharing runtime geometry, with OE-specific ranges."""

from pathlib import Path
from hashlib import sha256
import cv2
import numpy as np
from utils.py_utils.assets import sha256_file
from utils.py_utils.yoloe26_geometry import prepare_rgb
from samples.vision.ultralytics_yolo.runtime.python.detect import (
    resize_with_transform,
)
from samples.vision.yoloe.runtime.python.cli import resolve_selection


def calibration_tensor(image, target, variant):
    """Return RGB F32 NCHW: X5 0..255 (Mapper normalization), S 0..1 (ONNX input)."""
    selection = resolve_selection(target, variant=variant)
    if (
        not isinstance(image, np.ndarray)
        or image.ndim != 3
        or image.shape[2] != 3
        or image.dtype != np.uint8
        or min(image.shape[:2]) <= 0
    ):
        raise ValueError("Calibration requires nonempty uint8 BGR HWC images.")
    if selection.variant.startswith("26"):
        return prepare_rgb(image)
    pixels, _ = resize_with_transform(image, (640, 640), 1)
    tensor = np.ascontiguousarray(
        pixels[..., ::-1].transpose(2, 0, 1)[None], dtype=np.float32
    )
    return tensor if selection.target == "x5" else tensor / 255


def select_images(root, count):
    """Evenly sample sorted relative paths, with a deterministic case-sensitive tie break."""
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise ValueError("sample_count must be a positive integer.")
    root = Path(root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Calibration image directory missing: {root}")
    paths = sorted(
        (
            p
            for p in root.rglob("*")
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
        ),
        key=lambda p: (str(p.relative_to(root)).casefold(), str(p.relative_to(root))),
    )
    if not paths:
        raise ValueError(f"No calibration images below {root}.")
    return [
        paths[int(i)]
        for i in np.linspace(0, len(paths) - 1, min(count, len(paths)), dtype=int)
    ]


def write_calibration(paths, destination, target, variant):
    """Write new raw RGB float32 files for X5, normalized .npy tensors for S."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    records = []
    for i, path in enumerate(paths):
        source_bytes = path.read_bytes()
        image = (
            cv2.imdecode(np.frombuffer(source_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
            if source_bytes
            else None
        )
        if image is None:
            raise ValueError(f"Unreadable calibration image: {path}")
        tensor = calibration_tensor(image, target, variant)
        name = f"{i:06d}.rgb" if target == "x5" else f"{i:06d}.npy"
        output = destination / name
        if target == "x5":
            tensor.tofile(output)
        else:
            np.save(output, tensor, allow_pickle=False)
        records.append(
            {
                "source": str(path.resolve()),
                "source_sha256": sha256(source_bytes).hexdigest(),
                "tensor": name,
                "tensor_sha256": sha256_file(output),
                "bytes": output.stat().st_size,
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "range": [float(tensor.min()), float(tensor.max())],
            }
        )
    return records
