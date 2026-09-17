#!/usr/bin/env python3
# Copyright (c) 2026 D-Robotics Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Prepare PaddleOCR calibration tensors for the committed OE recipes.

The detector and recognizer use different calibration domains.  Detector
calibration follows the RGB values consumed by the conversion YAML's
``data_mean_and_scale`` policy.  Recognition calibration follows the runtime
writer and stores RGB float32 values in ``[0, 1]``.  X5's ``hb_mapper``
workflow consumes raw float32 files; the S-series ``hb_compile`` workflow
consumes NumPy files.  The formats are deliberately selected by ``--target``
instead of guessed from a filename.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}


def prepare_tensor(image: np.ndarray, *, target: str, stage: str) -> np.ndarray:
    """Return one NCHW RGB float32 calibration tensor.

    ``stage='detector'`` produces raw RGB values because both committed
    detector YAMLs apply their mean/scale in the compiler.  ``stage='recognizer'``
    produces the ``[0, 1]`` values written by both Python runtimes.
    """

    if target not in {"x5", "s100"}:
        raise ValueError("target must be x5 or s100")
    if stage not in {"detector", "recognizer"}:
        raise ValueError("stage must be detector or recognizer")
    if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("calibration image must be a BGR HxWx3 array")
    if image.dtype != np.uint8:
        raise ValueError(f"calibration image must be uint8, got {image.dtype}")

    if stage == "detector":
        size = (640, 640)
        interpolation = cv2.INTER_LINEAR if target == "x5" else cv2.INTER_AREA
    else:
        size = (320, 48)
        interpolation = cv2.INTER_LINEAR
    resized = cv2.resize(image, size, interpolation=interpolation)
    rgb = resized[:, :, ::-1].astype(np.float32)
    if stage == "recognizer":
        rgb /= 255.0
    return np.ascontiguousarray(rgb.transpose(2, 0, 1)[None])


def _images(path: Path) -> Iterable[Path]:
    if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
        return (path,)
    if not path.is_dir():
        return ()
    return tuple(
        item for item in sorted(path.rglob("*")) if item.suffix.lower() in IMAGE_SUFFIXES
    )


def main(argv: list[str] | None = None) -> int:
    """Read images and write calibration files without invoking the toolchain."""

    parser = argparse.ArgumentParser(
        description="Prepare PaddleOCR RGB float32 calibration files."
    )
    parser.add_argument(
        "--images", required=True, type=Path,
        help="Image file or directory containing calibration images.",
    )
    parser.add_argument(
        "--output", required=True, type=Path,
        help="Calibration directory consumed by the selected YAML.",
    )
    parser.add_argument(
        "--target", choices=("x5", "s100"), required=True,
        help="Compiler target; selects .rgbchw or .npy output.",
    )
    parser.add_argument(
        "--stage", choices=("detector", "recognizer"), required=True,
        help="OCR stage and input resolution/domain.",
    )
    parser.add_argument(
        "--max-images", type=int, default=50,
        help="Maximum number of images to convert (default: 50).",
    )
    args = parser.parse_args(argv)
    if args.max_images <= 0:
        parser.error("--max-images must be positive")
    args.images = args.images.expanduser()
    args.output = args.output.expanduser()

    candidates = tuple(_images(args.images))[: args.max_images]
    if not candidates:
        raise SystemExit(f"No supported images found under {args.images}")
    _prepare_output(args.output)
    written = 0
    for index, image_path in enumerate(candidates):
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            print(f"warning: skipped unreadable image {image_path}")
            continue
        tensor = prepare_tensor(image, target=args.target, stage=args.stage)
        stem = f"cal_{index:05d}"
        if args.target == "x5":
            destination = args.output / f"{args.target}_{args.stage}_{stem}.rgbchw"
            tensor.tofile(destination)
        else:
            destination = args.output / f"{args.target}_{args.stage}_{stem}.npy"
            np.save(destination, tensor)
        written += 1
    if not written:
        raise SystemExit("No readable calibration images were found")
    print(f"wrote {written} {args.target} {args.stage} calibration tensor(s) to {args.output}")
    return 0


def _prepare_output(output: Path) -> None:
    """Create an empty output directory to prevent mixed calibration tensors."""

    output = output.expanduser()
    output.mkdir(parents=True, exist_ok=True)
    entries = tuple(output.iterdir())
    if entries:
        raise SystemExit(
            f"calibration output must be empty to prevent mixed tensors: {output}; "
            "choose a new directory"
        )


if __name__ == "__main__":
    raise SystemExit(main())
