"""Command-line entry point for the YOLO26 Depth RDK X5 sample."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from yolo26_depth import Yolo26Depth, colorize_depth


def sha256(path: Path) -> str:
    """Calculate a SHA256 digest for a local file.

    Args:
        path: File to hash.

    Returns:
        Lowercase hexadecimal SHA256 digest.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    """Parse command-line options and run one depth inference."""
    parser = argparse.ArgumentParser(description="Run YOLO26 monocular depth on RDK X5.")
    parser.add_argument("--model", required=True, type=Path, help="X5 bayes-e .bin model")
    parser.add_argument("--input", required=True, type=Path, help="input image")
    parser.add_argument("--output", required=True, type=Path, help="new output directory")
    parser.add_argument("--warmup", type=int, default=3)
    args = parser.parse_args()

    if args.warmup < 0:
        raise ValueError("--warmup must be non-negative")
    if not args.model.is_file():
        raise FileNotFoundError(args.model)
    if not args.input.is_file():
        raise FileNotFoundError(args.input)
    args.output.mkdir(parents=True, exist_ok=True)

    image = cv2.imread(str(args.input), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"failed to decode {args.input}")

    estimator = Yolo26Depth(args.model)
    result = estimator.infer(image, warmup=args.warmup)
    depth_color = colorize_depth(result.depth_native)
    overlay = cv2.addWeighted(image, 0.45, depth_color, 0.55, 0.0)

    np.save(args.output / "log_depth.npy", result.log_depth)
    np.save(args.output / "depth_native.npy", result.depth_native)
    if not cv2.imwrite(str(args.output / "depth.png"), depth_color):
        raise OSError("failed to write depth.png")
    if not cv2.imwrite(str(args.output / "overlay.png"), overlay):
        raise OSError("failed to write overlay.png")

    report = {
        "schema_version": "1.0",
        "model": str(args.model),
        "model_sha256": sha256(args.model),
        "input": str(args.input),
        "input_sha256": sha256(args.input),
        "model_name": estimator.model_name,
        "input_name": estimator.input_name,
        "input_size": estimator.input_size,
        "output_name": estimator.output_name,
        "log_depth_shape": list(result.log_depth.shape),
        "depth_native_shape": list(result.depth_native.shape),
        "latency_ms": result.latency_ms,
        "geometry": result.geometry.__dict__,
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
