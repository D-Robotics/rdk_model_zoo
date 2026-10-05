# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""DINOv2 CLI surface: option declarations, summaries and evidence IO.

``main.py`` stays a thin entry that constructs the embedding task and calls
``predict``; the parser, the listing/dry-run modes, feature summaries and the
optional tensor export live here.  Nothing in this module embeds images or
loads a board SDK.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from samples.vision.dinov2.runtime.python.model_binding import (
    OUTPUT_SHAPES,
    SAMPLE_DIR,
    SUPPORTED_TARGETS,
    list_available_assets,
)

DEFAULT_TEST_IMAGE = SAMPLE_DIR / "test_data/dog.jpg"
DEFAULT_SECOND_IMAGE = SAMPLE_DIR / "test_data/bus.jpg"


def build_parser() -> argparse.ArgumentParser:
    """Build the parser without reading board identity or importing SDKs."""

    parser = argparse.ArgumentParser(description="DINOv2 ViT-S/14 image embedding")
    parser.add_argument("--target", choices=("auto", *SUPPORTED_TARGETS), default="auto", help="Concrete execution target.")
    parser.add_argument("--asset-id", default=None, help="Exact qualified manifest asset reference.")
    parser.add_argument("--model-path", default=None, help="Explicit local HBM path; requires --asset-id.")
    parser.add_argument("--test-img", default=str(DEFAULT_TEST_IMAGE), help="First BGR image path.")
    parser.add_argument("--second-img", default=str(DEFAULT_SECOND_IMAGE), help="Optional second image for cosine similarity.")
    parser.add_argument("--output", choices=("cls_feat", "patch_feat"), default="cls_feat", help="Feature output to return.")
    parser.add_argument("--priority", type=int, default=0, help="Runtime priority 0..255.")
    parser.add_argument("--bpu-cores", type=int, nargs="+", default=[0], help="Nonnegative BPU core indexes.")
    parser.add_argument("--output-file", default=None, help="Optional exact path for the returned NumPy tensor.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--list-models", action="store_true", help="List exact manifest assets without loading a model.")
    mode.add_argument("--dry-run", action="store_true", help="Resolve one target and print its source contract without SDK loading.")
    return parser


def run_list_models(target) -> int:
    """Print the exact manifest asset references for ``target`` (model-free)."""

    assets = list_available_assets(target)
    for asset in assets:
        print(asset.reference)
    print(f"{len(assets)} exact assets; supported targets: s100, s100p, s600; no model loaded.")
    return 0


def run_dry_run(selection) -> int:
    """Print the resolved selection contract without loading SDK or files."""

    print(json.dumps({
        "target": selection.target,
        "asset_id": selection.asset.reference,
        "model_path": str(selection.model_path),
        "model_path_exists": selection.model_path.is_file(),
        "input": {"name": "input", "shape": [1, 3, 224, 224], "dtype": "float32"},
        "outputs": {name: {"shape": list(shape), "transform": "runtime metadata required: F32=raw_f32; integer+quant=dequant"} for name, shape in OUTPUT_SHAPES.items()},
        "source_manifest": selection.asset.source_path,
    }, indent=2, ensure_ascii=False))
    return 0


def summary(tensor, output: str) -> dict:
    """Summarize one feature tensor without writing files."""

    flat = tensor.reshape(-1).astype("float32")
    return {
        "output": output,
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "mean": float(flat.mean()),
        "std": float(flat.std()),
        "min": float(flat.min()),
        "max": float(flat.max()),
        "l2_norm": float((flat ** 2).sum() ** 0.5),
    }


def cosine(first, second):
    """Cosine similarity of two flat feature tensors; ``None`` on zero norm."""

    import numpy as np

    a = first.reshape(-1).astype(np.float64)
    b = second.reshape(-1).astype(np.float64)
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a == 0.0 or norm_b == 0.0:
        return None
    return float(a @ b / (norm_a * norm_b))


def read_bgr_image(path):
    """Read one BGR image; decode failures name the exact path."""

    import cv2

    image = cv2.imread(str(Path(path).expanduser()), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot read image: {path}")
    return image


def save_feature(path: Path, feature) -> None:
    """Write one feature tensor to an exact path, creating parents."""

    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        np.save(handle, feature, allow_pickle=False)
    print(f"Feature tensor saved: {path}")
