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
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import sys

from utils.py_utils.assets import Asset, list_assets
from utils.py_utils.platforms import resolve_target

SAMPLE_DIR = Path(__file__).resolve().parents[2]
SUPPORTED_TARGETS = ("s100", "s100p", "s600")
MARCHES = {"s100": "nash-e", "s100p": "nash-m", "s600": "nash-p"}
OUTPUTS = ("cls_feat", "patch_feat")
OUTPUT_SHAPES = {
    "cls_feat": (1, 384),
    "patch_feat": (1, 256, 384),
}

DEFAULT_TEST_IMAGE = SAMPLE_DIR / "test_data/dog.jpg"
DEFAULT_SECOND_IMAGE = SAMPLE_DIR / "test_data/bus.jpg"


@dataclass(frozen=True)
class ModelSelection:
    """One published manifest asset and its selected local path.

    Attributes:
        asset: Manifest asset record backing the selection.
        target: Concrete execution target (s100, s100p, or s600).
        model_path: Local compiled model path.
        explicit_model_path: Whether the caller supplied the path explicitly.
    """

    asset: Asset
    target: str
    model_path: Path
    explicit_model_path: bool = False


def _expected_filename(target: str) -> str:
    march = MARCHES[target]
    suffix = {"nash-e": "nashe", "nash-m": "nashm", "nash-p": "nashp"}[march]
    return f"{march}/dinov2_vits14_224_int16_{suffix}.hbm"


def list_available_assets(target: Optional[str] = None) -> tuple[Asset, ...]:
    """Return the three exact HBM rows from the S publication manifest.

    Args:
        target: Concrete target filter; ``auto``/None lists all publications.

    Returns:
        tuple[Asset, ...]: Published assets in manifest order.

    Raises:
        ValueError: The target is unknown or the publication changed.
    """
    if target not in (None, "auto", *SUPPORTED_TARGETS):
        if target in ("x5",):
            return ()
        raise ValueError(f"Unknown target: {target}")
    assets = list_assets("s", "dinov2")
    expected = {_expected_filename(key) for key in SUPPORTED_TARGETS}
    actual = {asset.filename for asset in assets}
    if actual != expected or any(asset.format != "hbm" for asset in assets):
        raise ValueError("DINOv2 publication changed; review its finite contracts first.")
    if target in (None, "auto"):
        return tuple(asset for asset in assets if asset.filename in expected)
    filename = _expected_filename(target)
    return tuple(asset for asset in assets if asset.filename == filename)


def resolve_selection(
    target: str = "auto",
    *,
    asset_id: Optional[str] = None,
    model_path: "str | Path | None" = None,
    soc_name: Optional[str] = None,
    board_type: Optional[str] = None,
) -> ModelSelection:
    """Resolve an exact published asset without filename or target fallback.

    Args:
        target: ``auto`` resolves the executing board; s100/s100p/s600 are
            the published targets.
        asset_id: Qualified manifest reference; a ``model_path`` override
            requires the exact reference.
        model_path: Optional explicit local path for the selected asset.
        soc_name: Optional board-identity override for ``auto`` resolution.
        board_type: Optional board-type override for ``auto`` resolution.

    Returns:
        ModelSelection: Concrete target, manifest asset, and local path.

    Raises:
        ValueError: The target, asset, or path combination is invalid.
    """
    resolved = resolve_target(target, soc_name=soc_name, board_type=board_type)
    if resolved not in SUPPORTED_TARGETS:
        raise ValueError(f"No published DINOv2 support for {resolved}.")
    if model_path is not None and asset_id is None:
        raise ValueError("An external model-path requires the exact manifest asset-id.")

    assets = list_available_assets(resolved)
    if asset_id is None:
        matches = assets
    else:
        matches = tuple(asset for asset in assets if asset.reference == asset_id)
    if len(matches) != 1:
        available = ", ".join(asset.reference for asset in assets)
        raise ValueError(f"Unknown DINOv2 asset-id {asset_id!r}; available: {available}")
    asset = matches[0]
    path = Path(model_path).expanduser() if model_path is not None else SAMPLE_DIR / "model" / asset.filename
    return ModelSelection(asset, resolved, path, model_path is not None)


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
