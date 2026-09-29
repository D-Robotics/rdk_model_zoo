# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Single-image relative-depth numerical comparison and shared-range rendering."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
from samples._shared.assets import sha256_file
from samples.vision.yolo26_depth.evaluator.metrics import fidelity_metrics


def colorize(depth: np.ndarray, low: float, high: float) -> np.ndarray:
    """Colorize depth with one shared range for visual comparison.

    Args:
        depth: Depth map to visualize.
        low: Lower visualization bound.
        high: Upper visualization bound.

    Returns:
        BGR Turbo visualization.
    """
    normalized = np.clip((depth - low) / max(high - low, 1e-6), 0.0, 1.0)
    gray = np.asarray(normalized * 255.0, dtype=np.uint8)
    return cv2.applyColorMap(255 - gray, cv2.COLORMAP_TURBO)


def fit(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """Fit one visualization into a fixed contact-sheet cell.

    Args:
        image: BGR image to place in the cell.
        width: Target cell width.
        height: Target cell height.

    Returns:
        Letterboxed BGR cell image.
    """
    source_height, source_width = image.shape[:2]
    ratio = min(width / source_width, height / source_height)
    resized_width = max(1, round(source_width * ratio))
    resized_height = max(1, round(source_height * ratio))
    resized = cv2.resize(
        image, (resized_width, resized_height), interpolation=cv2.INTER_AREA
    )
    canvas = np.full((height, width, 3), 32, dtype=np.uint8)
    left = (width - resized_width) // 2
    top = (height - resized_height) // 2
    canvas[top : top + resized_height, left : left + resized_width] = resized
    return canvas


def fidelity(candidate, reference):
    result = fidelity_metrics([candidate], [reference])
    ratio = np.maximum(candidate / reference, reference / candidate)
    result["delta1"] = float((ratio < 1.25).mean())
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--image", type=Path, required=True)
    p.add_argument(
        "--official",
        type=Path,
        required=True,
        help="Caller-provided restored floating reference; not automatically certified official",
    )
    p.add_argument("--candidate", "--x5", dest="candidate", type=Path, required=True)
    p.add_argument("--candidate-name", default="caller-provided candidate")
    p.add_argument("--reference-name", default="caller-provided floating reference")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(args.output)
    image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    reference = np.load(args.official, allow_pickle=False).astype(np.float32)
    candidate = np.load(args.candidate, allow_pickle=False).astype(np.float32)
    if (
        image is None
        or reference.ndim != 2
        or reference.shape != candidate.shape
        or reference.shape != image.shape[:2]
    ):
        raise ValueError(
            "Reference/candidate must have the same HxW as the decoded image"
        )
    if (
        not np.isfinite(reference).all()
        or not np.isfinite(candidate).all()
        or np.any(reference <= 1e-6)
        or np.any(candidate <= 1e-6)
    ):
        raise ValueError(
            "Comparison requires finite strictly positive relative-depth maps"
        )
    # This tool preserves the source NumPy average-of-middle median. SUNRGBD
    # aggregation separately preserves the source Torch lower-median policy.
    scale = float(np.median(reference) / np.median(candidate))
    aligned = candidate * scale
    low, high = np.percentile(reference, [2.0, 98.0])
    ref_color = colorize(reference, float(low), float(high))
    candidate_color = colorize(aligned, float(low), float(high))
    relative_error = np.abs(aligned - reference) / reference
    error_gray = (np.clip(relative_error / 0.5, 0, 1) * 255).astype(np.uint8)
    error_color = cv2.applyColorMap(error_gray, cv2.COLORMAP_INFERNO)
    images = {
        "reference_depth_common_range.png": ref_color,
        "candidate_depth_median_aligned_common_range.png": candidate_color,
        "absolute_relative_error.png": error_color,
        "reference_overlay_common_range.png": cv2.addWeighted(
            image, 0.45, ref_color, 0.55, 0
        ),
        "candidate_overlay_median_aligned_common_range.png": cv2.addWeighted(
            image, 0.45, candidate_color, 0.55, 0
        ),
    }
    width, height, header = 405, 540, 46
    sheet = np.full((height + header, width * 4, 3), 245, np.uint8)
    for index, (title, visual) in enumerate(
        zip(
            (
                "Input",
                "Reference",
                "Candidate (median-aligned)",
                "Absolute relative error",
            ),
            (image, ref_color, candidate_color, error_color),
        )
    ):
        left = index * width
        cv2.putText(
            sheet,
            title,
            (left + 10, 31),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (20, 20, 20),
            2,
            cv2.LINE_AA,
        )
        sheet[header:, left : left + width] = fit(visual, width, height)
    images["comparison.jpg"] = sheet
    report = {
        "schema_version": "2.0",
        "reference": args.reference_name,
        "candidate": args.candidate_name,
        "image": str(args.image),
        "image_sha256": sha256_file(args.image),
        "reference_output": str(args.official),
        "reference_output_sha256": sha256_file(args.official),
        "candidate_output": str(args.candidate),
        "candidate_output_sha256": sha256_file(args.candidate),
        "shape": list(reference.shape),
        "raw": fidelity(candidate, reference),
        "median_scale_candidate_to_reference": scale,
        "median_policy": "NumPy midpoint average, matching source single-image tool",
        "median_aligned": fidelity(aligned, reference),
        "common_color_range_percentile_2_98": [float(low), float(high)],
        "board": "not-run by this offline tool",
        "dataset_accuracy": "not measured",
    }
    args.output.mkdir(parents=True, exist_ok=False)
    np.save(
        args.output / "candidate_depth_median_aligned.npy", aligned, allow_pickle=False
    )
    for name, value in images.items():
        if not cv2.imwrite(str(args.output / name), value):
            raise OSError(f"Could not write {args.output/name}")
    text = json.dumps(report, indent=2, allow_nan=False)
    (args.output / "comparison-report.json").write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
