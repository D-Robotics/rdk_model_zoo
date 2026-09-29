# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Resolve a published YOLOv5 asset before invoking the native executable."""

from __future__ import annotations

import argparse
import importlib
import json
import math
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
CPP = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
binding = importlib.import_module("samples.vision.yolov5.runtime.python.model_binding")

# The fixed X5 C++ source (runtime/cpp/main.cc MODEL_PATH) defaults to the
# s-v2.0 artifact, while the unified Python runtime defaults to n-v7.0. Neither
# choice is a substitute for the other, so the native launcher keeps the C++
# source default when the caller names no variant and no asset id.
X5_CPP_SOURCE_DEFAULT_VARIANT = "s-v2.0"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=("auto", "x5", "s100", "s100p", "s600"), default="auto")
    parser.add_argument("--variant")
    parser.add_argument("--asset-id")
    parser.add_argument("--model-path")
    parser.add_argument("--test-img")
    parser.add_argument("--label-file")
    parser.add_argument("--output", default="result.jpg")
    parser.add_argument("--dump-dir", type=Path)
    parser.add_argument("--score-thres", type=float, default=0.25)
    parser.add_argument("--nms-thres", type=float, default=0.45)
    parser.add_argument("--priority", type=int, default=0)
    parser.add_argument("--bpu-core", type=int, default=-1)
    parser.add_argument("--binary", type=Path)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--list-models", action="store_true")
    modes.add_argument("--dry-run", action="store_true")
    return parser


def _selection(args):
    variant = args.variant
    if args.target == "x5" and variant is None and args.asset_id is None:
        variant = X5_CPP_SOURCE_DEFAULT_VARIANT
    return binding.resolve_selection(
        args.target, variant=variant, asset_id=args.asset_id, model_path=args.model_path
    )


def _validate_runtime_args(args) -> None:
    if not all(math.isfinite(value) and 0.0 <= value <= 1.0
               for value in (args.score_thres, args.nms_thres)):
        raise ValueError("thresholds must be finite values in [0,1]")
    if not 0 <= args.priority <= 255:
        raise ValueError("priority must be between 0 and 255")
    if args.bpu_core < -1:
        raise ValueError("bpu-core must be -1 or a non-negative index")
    if args.target == "x5" and (args.priority != 0 or args.bpu_core != -1):
        raise ValueError(
            "x5 has no verified HB-DNN scheduling mapping; only --priority 0 "
            "--bpu-core -1 are accepted"
        )


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.list_models:
            for asset in binding.list_available_assets(args.target):
                print(asset.reference)
            return 0
        if args.dry_run and args.target == "auto":
            raise ValueError("--dry-run requires an explicit --target")
        _validate_runtime_args(args)
        selection = _selection(args)
        image = Path(args.test_img).expanduser() if args.test_img else (
            binding.SAMPLE_DIR / "test_data" / ("bus.jpg" if selection.target == "x5" else "kite.jpg")
        )
        payload = {
            "target": selection.target,
            "variant": selection.variant,
            "asset_id": selection.asset.reference,
            "filename": selection.asset.filename,
            "model_path": str(selection.model_path),
            "test_img": str(image),
            "board_status": "not-run",
        }
        if args.dry_run:
            print(json.dumps(payload, indent=2))
            return 0
        if not selection.model_path.is_file():
            raise FileNotFoundError(f"Resolved model path does not exist: {selection.model_path}")
        from samples._shared.assets import verify_asset_file
        verify_asset_file(selection.asset, selection.model_path)
        from samples._shared.platforms import require_execution_target
        require_execution_target(selection.target)
        binary = args.binary or (CPP / "build" / selection.target / "yolov5_cpp")
        if not binary.is_file():
            raise FileNotFoundError(f"Native executable does not exist: {binary}")
        command = [str(binary), "--target", selection.target, "--model-path", str(selection.model_path),
                   "--test-img", str(image), "--output", args.output,
                   "--score-thres", str(args.score_thres), "--nms-thres", str(args.nms_thres),
                   "--priority", str(args.priority), "--bpu-core", str(args.bpu_core),
                   "--asset-id", selection.asset.reference]
        if args.label_file:
            command.extend(["--label-file", args.label_file])
        if args.dump_dir is not None:
            command.extend(["--dump-dir", str(args.dump_dir)])
        return subprocess.call(command)
    except (ValueError, OSError, RuntimeError, ImportError) as error:
        print(f"yolov5_cpp launcher: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
