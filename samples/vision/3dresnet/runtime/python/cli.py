# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""R3D-18 CLI surface: option declarations and the model-free/report helpers.

``main.py`` stays a thin entry that constructs the video task and calls
``predict``; the parser, the listing/dry-run modes and the JSON report
assembly live here.  Nothing in this module classifies clips or loads a board
SDK.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .model_binding import (
    CLASS_COUNT,
    INPUT_SHAPE,
    SAMPLE_DIR,
    SUPPORTED_TARGETS,
    list_available_assets,
)

DEFAULT_CLIP = SAMPLE_DIR / "test_data/video0.npy"
DEFAULT_LABELS = SAMPLE_DIR / "test_data/kinetics_classnames.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run 3D ResNet-18 video action classification.")
    parser.add_argument("--target", choices=("auto", *SUPPORTED_TARGETS), default="auto")
    parser.add_argument("--asset-id", default=None, help="Exact manifest asset reference.")
    parser.add_argument("--model-path", default=None, help="External HBM path; requires --asset-id.")
    parser.add_argument("--test-clip", default=str(DEFAULT_CLIP), help="Preprocessed float32 clip .npy path.")
    parser.add_argument("--label-file", default=str(DEFAULT_LABELS), help="Kinetics-400 JSON mapping path.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of predictions, 1..400.")
    parser.add_argument("--priority", type=int, default=0, help="Runtime priority 0..255.")
    parser.add_argument("--bpu-cores", type=int, nargs="+", default=[0], help="Nonnegative BPU core indexes.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--list-models", action="store_true", help="List exact manifest assets without loading SDK.")
    mode.add_argument("--dry-run", action="store_true", help="Resolve the contract without loading SDK or files.")
    return parser


def parse_args(argv=None) -> argparse.Namespace:
    """Parse the stable CLI contract without reading board identity."""

    return build_parser().parse_args(argv)


def run_list_models(target) -> int:
    """Print the exact manifest asset references for ``target`` (model-free)."""

    assets = list_available_assets(target)
    for asset in assets:
        print(asset.reference)
    print(f"{len(assets)} exact asset; no model loaded.")
    return 0


def run_dry_run(selection) -> int:
    """Print the resolved selection contract without loading SDK or files."""

    print(json.dumps({
        "target": selection.target,
        "asset_id": selection.asset.reference,
        "model_path": str(selection.model_path),
        "model_path_exists": selection.model_path.is_file(),
        "input": {"name": "runtime metadata input name", "shape": list(INPUT_SHAPE), "dtype": "float32"},
        "output": {"name": "runtime metadata output name", "shape": [1, CLASS_COUNT], "dtype": "float32", "semantic": "source logits"},
        "source_manifest": selection.asset.source_path,
    }, indent=2, ensure_ascii=False))
    return 0


def report(result, labels: dict[int, str], selection, clip_path: Path) -> dict:
    """Assemble the JSON report for one finished prediction."""

    return {
        "asset_id": selection.asset.reference,
        "target": selection.target,
        "clip": str(clip_path),
        "predictions": [
            {"class_id": int(class_id), "score": float(score), "label": labels.get(int(class_id), str(int(class_id)))}
            for class_id, score in zip(result.class_ids, result.scores)
        ],
    }
