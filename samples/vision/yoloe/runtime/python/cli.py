# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""YOLOE CLI surface: option declarations, inspection modes and result display.

``main.py`` stays a thin entry that constructs the model and calls ``predict``;
the parser, the model-free listing/dry-run modes and the result report live
here.  Nothing in this module runs segmentation or loads a board SDK.
"""

from dataclasses import asdict
import argparse
import json
from pathlib import Path
import sys

from samples.vision.yoloe.runtime.python.model_binding import (
    SAMPLE_DIR,
    list_models,
)


def build_parser():
    p = argparse.ArgumentParser(
        description="YOLOE-11/26 prompt-free segmentation with floating model outputs."
    )
    p.add_argument(
        "--target", choices=("auto", "x5", "s100", "s100p", "s600"), default="auto"
    )
    p.add_argument(
        "--variant",
        default=None,
        help="11s/m/l or 26n/s/m/l/x; defaults x5/s100=11s, s100p=26n.",
    )
    p.add_argument(
        "--asset-id", default=None, help="Exact original publication identity."
    )
    p.add_argument("--model-path", default=None)
    p.add_argument(
        "--local-float-sha256",
        default=None,
        help="SHA-256 of your separately converted floating-output model; requires --model-path.",
    )
    p.add_argument("--test-img", default=str(SAMPLE_DIR / "test_data/office_desk.jpg"))
    p.add_argument("--label-file", default=str(SAMPLE_DIR / "test_data/classes.names"))
    p.add_argument("--img-save-path", default=str(SAMPLE_DIR / "test_data/result.jpg"))
    p.add_argument("--score-thres", type=float, default=0.25)
    p.add_argument(
        "--nms-thres",
        type=float,
        default=None,
        help="YOLOE-11 default .7; forbidden for 26.",
    )
    p.add_argument("--resize-type", type=int, choices=(0, 1), default=1)
    p.add_argument(
        "--no-morph",
        action="store_true",
        help="Disable source CLI default opening for S YOLOE-11 ROI masks.",
    )
    p.add_argument(
        "--no-contour", action="store_true", help="Disable mask contour outlines."
    )
    p.add_argument("--max-det", type=int, default=300, help="YOLOE-26 only, 1..8400.")
    p.add_argument(
        "--multi-label",
        action="store_true",
        help="YOLOE-26 only; permit multiple classes per anchor.",
    )
    p.add_argument("--priority", type=int, default=0)
    p.add_argument("--bpu-cores", type=int, nargs="+", default=[0])
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--list-models", action="store_true")
    mode.add_argument("--dry-run", action="store_true")
    return p


def run_list_models(target) -> int:
    """Print the manifest model matrix for ``target`` (model-free)."""

    print(
        json.dumps(
            [
                {
                    "target": t,
                    "variant": v,
                    "asset_id": a.reference,
                    "published_float": t == "x5",
                }
                for t, v, a in list_models(target)
            ],
            indent=2,
        )
    )
    return 0


def run_dry_run(selection, config) -> int:
    """Print the resolved selection and config without loading a model."""

    print(
        json.dumps(
            {
                "target": selection.target,
                "variant": selection.variant,
                "source_asset_id": selection.asset.reference,
                "model_path": str(selection.model_path),
                "published_float": selection.published_float,
                "local_float_sha256": selection.local_float_sha256,
                "runtime_metadata_verified": False,
                "config": asdict(config),
                "status": (
                    "metadata validation required"
                    if selection.published_float or selection.local_float
                    else "requires separately converted floating-output model"
                ),
            },
            indent=2,
        )
    )
    return 0


def validate_scheduling(args) -> None:
    """Reject scheduling values the run would refuse, including during dry-run."""

    if (
        not 0 <= args.priority <= 255
        or not args.bpu_cores
        or any(c < 0 for c in args.bpu_cores)
    ):
        raise ValueError("Invalid scheduling priority/core list.")


def print_result_report(result, selection, args) -> None:
    """Print the JSON detection summary for one finished prediction."""

    print(
        json.dumps(
            {
                "target": selection.target,
                "variant": selection.variant,
                "count": len(result.scores),
                "class_ids": result.class_ids.tolist(),
                "scores": result.scores.tolist(),
                "mask_layout": result.mask_layout,
                "image_saved": str(Path(args.img_save_path).expanduser()),
            },
            indent=2,
        )
    )
