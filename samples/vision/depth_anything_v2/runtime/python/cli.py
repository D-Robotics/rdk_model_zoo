# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Depth CLI surface: option declarations, path checks, reports and file IO.

``main.py`` stays a thin entry that constructs the task and calls ``predict``;
everything presentational — the parser, the model-free listing/dry-run modes,
destination validation and the canonical output/report writing — lives here.
Nothing in this module runs inference.
"""

import argparse
import json
from pathlib import Path
import sys

from samples.vision.depth_anything_v2.runtime.python.model_binding import (
    SAMPLE_DIR,
    list_available_assets,
)

#: Canonical files an additional ``--img-save-path`` may never replace.
CANONICAL_OUTPUTS = (
    "raw_depth.npy",
    "depth_native.npy",
    "depth_gray.png",
    "depth_color.png",
    "report.json",
)


def build_parser():
    p = argparse.ArgumentParser(
        description="Depth Anything V2 relative depth; published S100 artifact only"
    )
    p.add_argument(
        "--target", choices=("auto", "x5", "s100", "s100p", "s600"), default="auto"
    )
    p.add_argument("--asset-id")
    p.add_argument("--model-path", type=Path)
    p.add_argument(
        "--test-img", type=Path, default=SAMPLE_DIR / "test_data/furseal.jpg"
    )
    p.add_argument("--output", type=Path, default=Path("outputs/depth_anything_v2"))
    p.add_argument(
        "--img-save-path",
        type=Path,
        help="Optional additional source-compatible color image path; must not exist",
    )
    p.add_argument("--resize-type", type=int, choices=(0, 1), default=0)
    p.add_argument("--priority", type=int, default=0)
    p.add_argument("--bpu-cores", type=int, nargs="+", default=[0])
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--list-models", action="store_true")
    mode.add_argument("--dry-run", action="store_true")
    return p


def run_list_models(target) -> int:
    """Print the manifest-backed assets for ``target`` (model-free)."""

    print(
        json.dumps(
            [
                {"asset_id": a.reference, "url": a.url, "sha256": a.sha256}
                for a in list_available_assets(target)
            ],
            indent=2,
        )
    )
    return 0


def run_dry_run(selection, args) -> int:
    """Print the resolved selection without loading a model or SDK."""

    print(
        json.dumps(
            {
                "target": selection.target,
                "asset_id": selection.asset.reference,
                "model_path": str(selection.model_path),
                "resize_type": args.resize_type,
                "sdk_loaded": False,
                "downloaded": False,
            },
            indent=2,
        )
    )
    return 0


def validate_output_paths(output: Path, extra: Path | None) -> None:
    """Refuse existing outputs and additional paths over canonical files."""

    if output.exists():
        raise FileExistsError(f"Use a new output directory: {output}")
    if extra is not None and extra.resolve() in {
        (output / name).resolve() for name in CANONICAL_OUTPUTS
    }:
        raise ValueError("Additional color image must not replace a canonical output")
    if extra is not None and extra.exists():
        raise FileExistsError(f"Use a new image path: {extra}")


def read_bgr_image(path: Path):
    """Read one BGR image; decode failures name the exact path."""

    import cv2

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot decode input image: {path}")
    return image


def save_depth_evidence(
    output: Path,
    extra: Path | None,
    *,
    selection,
    runner,
    binding,
    args,
    image_path: Path,
    details,
) -> None:
    """Write the canonical raw/native tensors, displays and provenance report."""

    import cv2
    import numpy as np

    from samples._shared.assets import sha256_file
    from samples._shared.runtime_meta import metadata_evidence
    from samples.vision.depth_anything_v2.runtime.python.visualization import (
        normalize_depth,
        colorize_depth,
    )

    result = details.result
    color = colorize_depth(result.depth_native)
    report = {
        "schema_version": "1.0",
        "target": selection.target,
        "asset_id": selection.asset.reference,
        "model_path": str(selection.model_path),
        "model_sha256": sha256_file(selection.model_path),
        "publisher_sha256": selection.asset.sha256,
        "input": str(image_path),
        "input_sha256": sha256_file(image_path),
        "runtime_version": str(getattr(runner.runtime, "version", "unknown")),
        "runtime_metadata": metadata_evidence(binding.metadata),
        "resize_type": args.resize_type,
        "input_normalization": "pixelwise RGB z-score, epsilon=1e-5",
        "output_units": "relative, not meters",
        "depth_native_shape": list(result.depth_native.shape),
        "priority": args.priority,
        "bpu_cores": args.bpu_cores,
        "restoration": "OpenCV INTER_LINEAR; optional letterbox crop before original-size resize",
        "constant_visualization": "all zero grayscale",
        "latency": "not measured",
        "additional_color_image": str(extra) if extra else None,
    }
    output.mkdir(parents=True, exist_ok=False)
    np.save(output / "raw_depth.npy", details.raw)
    np.save(output / "depth_native.npy", result.depth_native)
    if not cv2.imwrite(
        str(output / "depth_gray.png"), normalize_depth(result.depth_native)
    ) or not cv2.imwrite(str(output / "depth_color.png"), color):
        raise OSError("Failed to write depth visualization")
    if extra is not None:
        extra.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(extra), color):
            raise OSError(f"Failed to write {extra}")
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
