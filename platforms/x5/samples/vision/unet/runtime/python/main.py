#!/usr/bin/env python3
# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0

"""Run one UNet image on RDK X5 and save mask visualizations."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from unet import DEFAULT_MODEL_PATH, UNet, UNetConfig, colorize_mask

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_TEST_IMAGE = SCRIPT_DIR.parent.parent / "test_data" / "2007_000033.jpg"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments with repository-standard defaults.

    Returns:
        Parsed command-line namespace.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path to the X5 UNet *.bin model.",
    )
    parser.add_argument(
        "--test-img",
        type=Path,
        default=DEFAULT_TEST_IMAGE,
        help="Path to one input image readable by OpenCV.",
    )
    parser.add_argument(
        "--mask-save-path",
        type=Path,
        default=Path("unet_mask.png"),
        help="Path for the raw class-index PNG mask.",
    )
    parser.add_argument(
        "--img-save-path",
        type=Path,
        default=Path("unet_result.png"),
        help="Path for the color overlay image.",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("unet_runtime_report.json"),
        help="Path for the machine-readable runtime report.",
    )
    parser.add_argument(
        "--priority",
        type=int,
        default=None,
        help="Optional BPU scheduling priority; omitted by default.",
    )
    parser.add_argument(
        "--bpu-core",
        type=int,
        default=None,
        help="Optional BPU core index; omitted for automatic scheduling.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.55,
        help="Segmentation color weight in the output overlay.",
    )
    return parser.parse_args()


def main() -> int:
    """Load an image, run UNet, and persist reproducible outputs.

    Returns:
        Process exit status, zero on success.

    Raises:
        FileNotFoundError: If the input image cannot be loaded.
        ValueError: If the overlay alpha is outside ``[0, 1]``.
    """

    args = parse_args()
    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError("--alpha must be between 0 and 1")
    image_path = args.test_img.expanduser().resolve()
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"failed to read image: {image_path}")

    config = UNetConfig(model_path=str(args.model_path.expanduser().resolve()))
    model = UNet(config)
    cores = [args.bpu_core] if args.bpu_core is not None else None
    model.set_scheduling_params(priority=args.priority, bpu_cores=cores)

    start = time.perf_counter()
    mask = model.predict(image)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    color_mask = colorize_mask(mask, config.num_classes)
    resized = cv2.resize(
        image,
        (config.input_width, config.input_height),
        interpolation=cv2.INTER_LINEAR,
    )
    overlay = cv2.addWeighted(resized, 1.0 - args.alpha, color_mask, args.alpha, 0.0)

    mask_path = args.mask_save_path.expanduser().resolve()
    image_output_path = args.img_save_path.expanduser().resolve()
    report_path = args.report_path.expanduser().resolve()
    for output_path in (mask_path, image_output_path, report_path):
        output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(mask_path), mask):
        raise OSError(f"failed to write mask: {mask_path}")
    if not cv2.imwrite(str(image_output_path), overlay):
        raise OSError(f"failed to write overlay: {image_output_path}")

    report = {
        "model_path": str(Path(config.model_path).resolve()),
        "image_path": str(image_path),
        "runtime_version": model.model.version,
        "model_name": model.model_name,
        "input": {
            "name": model.input_name,
            "shape": list(model.input_shape),
            "dtype": model.input_dtype,
        },
        "output": {
            "name": model.output_name,
            "shape": list(model.output_shape),
            "dtype": model.output_dtype,
        },
        "mask_shape": list(mask.shape),
        "classes_present": np.unique(mask).astype(int).tolist(),
        "elapsed_ms": elapsed_ms,
        "mask_save_path": str(mask_path),
        "img_save_path": str(image_output_path),
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
