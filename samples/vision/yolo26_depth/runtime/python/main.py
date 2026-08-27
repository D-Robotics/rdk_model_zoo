"""Command-line entry point for the YOLO26 Depth RDK-S sample."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from yolo26_depth import Yolo26Depth, colorize_depth

REPOSITORY_ROOT = Path(__file__).resolve().parents[5]
import sys
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.append(str(REPOSITORY_ROOT))
from utils.py_utils import inspect


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


def model_filename(variant: str, suffix: str) -> str:
    """Return the delivered HBM filename for one variant and march suffix.

    Args:
        variant: One of ``n`` / ``s`` / ``m`` / ``l`` / ``x``.
        suffix: March filename suffix (``nashe`` / ``nashm`` / ``nashp``).

    Returns:
        The model filename under ``model/<march>/``.
    """
    if variant in ("l", "x"):
        return f"yolo26{variant}_depth_lite_{suffix}_768x768.hbm"
    return f"yolo26{variant}_depth_{suffix}_768x768_nv12.hbm"


def main() -> None:
    """Parse command-line options and run one depth inference."""
    parser = argparse.ArgumentParser(description="Run YOLO26 monocular depth on RDK-S.")
    parser.add_argument("--model", type=Path, default=None, help="Override default S-series .hbm model")
    parser.add_argument("--variant", choices=("n", "s", "m", "l", "x"), default="n")
    parser.add_argument("--input", required=True, type=Path, help="input image")
    parser.add_argument("--output", required=True, type=Path, help="new output directory")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--priority", type=int, default=0, help="Model scheduling priority")
    parser.add_argument("--bpu-cores", nargs="+", type=int, default=[0], help="BPU core indexes")
    args = parser.parse_args()

    if args.warmup < 0:
        raise ValueError("--warmup must be non-negative")
    if args.model is None:
        board_type = ""
        try:
            board_type = Path("/sys/class/boardinfo/board_type").read_text(encoding="utf-8")
        except OSError:
            pass
        _, march, suffix, _ = inspect.resolve_platform(
            inspect.get_soc_name_fallback_free(), board_type
        )
        args.model = Path(__file__).resolve().parents[2] / "model" / march / model_filename(
            args.variant, suffix
        )
    if not args.model.is_file():
        raise FileNotFoundError(args.model)
    if not args.input.is_file():
        raise FileNotFoundError(args.input)
    args.output.mkdir(parents=True, exist_ok=True)

    image = cv2.imread(str(args.input), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"failed to decode {args.input}")

    estimator = Yolo26Depth(args.model, args.variant)
    estimator.set_scheduling_params(args.priority, args.bpu_cores)
    result = estimator.infer(image, warmup=args.warmup)
    depth_color = colorize_depth(result.depth_native)
    overlay = cv2.addWeighted(image, 0.45, depth_color, 0.55, 0.0)

    np.save(args.output / "log_depth.npy", result.log_depth)
    np.save(args.output / "depth_native.npy", result.depth_native)
    if result.raw_logit is not None:
        np.save(args.output / "raw_logit.npy", result.raw_logit)
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
        "profile": result.profile,
        "log_depth_shape": list(result.log_depth.shape),
        "depth_native_shape": list(result.depth_native.shape),
        "latency_ms": result.latency_ms,
    }
    if result.raw_logit is not None:
        report["raw_logit_shape"] = list(result.raw_logit.shape)
        report["calibration"] = {
            "cal_a": estimator.cal_a,
            "cal_b": estimator.cal_b,
            "clip": [-4.0, 5.0],
        }
    (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
