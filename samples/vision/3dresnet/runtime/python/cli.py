# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""R3D-18 CLI options, published-model selection, listing, dry-run, and report.

``main.py`` uses these helpers to parse arguments, preview a selection, load
the Kinetics-400 label mapping, and assemble the JSON prediction report. The
classification flow itself lives in ``classification.py``.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from utils.py_utils.assets import Asset, list_assets
from utils.py_utils.platforms import resolve_target

SAMPLE_DIR = Path(__file__).resolve().parents[2]
SUPPORTED_TARGETS = ("s100",)
ASSET_FILENAME = "s100/r3d_18.hbm"
ASSET_ID = "s:3dresnet:s100/r3d_18.hbm"
INPUT_SHAPE = (1, 3, 16, 112, 112)
CLASS_COUNT = 400

DEFAULT_CLIP = SAMPLE_DIR / "test_data/video0.npy"
DEFAULT_LABELS = SAMPLE_DIR / "test_data/kinetics_classnames.json"


@dataclass(frozen=True)
class ModelSelection:
    """One published manifest asset and its selected local path.

    Attributes:
        asset: Manifest asset record backing the selection.
        target: Concrete execution target (``s100``).
        model_path: Local compiled model path.
        explicit_model_path: Whether the caller supplied the path explicitly.
    """

    asset: Asset
    target: str
    model_path: Path
    explicit_model_path: bool = False


def _published_asset() -> Asset:
    assets = tuple(asset for asset in list_assets("s", "3dresnet") if asset.filename == ASSET_FILENAME)
    if len(assets) != 1 or assets[0].format != "hbm":
        raise ValueError("The published 3DResNet S100 HBM asset is missing or changed.")
    return assets[0]


def list_available_assets(target: Optional[str] = None) -> tuple[Asset, ...]:
    """List published R3D-18 assets for a target.

    Args:
        target: Concrete target filter; ``auto``/None resolves to S100.

    Returns:
        tuple[Asset, ...]: Published assets in manifest order.
    """
    if target not in (None, "auto", *SUPPORTED_TARGETS):
        return ()
    asset = _published_asset()
    if target in (None, "auto", "s100"):
        return (asset,)
    return ()


def resolve_selection(
    target: str = "auto",
    *,
    asset_id: Optional[str] = None,
    model_path: "str | Path | None" = None,
    soc_name: Optional[str] = None,
    board_type: Optional[str] = None,
) -> ModelSelection:
    """Resolve the published R3D-18 artifact for a command.

    Args:
        target: ``auto`` resolves the executing board; s100 is the only
            published target.
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
        raise ValueError(f"No published 3DResNet support for {resolved}.")
    if model_path is not None and asset_id is None:
        raise ValueError("An external model-path requires the exact manifest asset-id.")
    asset = _published_asset()
    if asset_id is not None and asset_id != asset.reference:
        raise ValueError(f"Unknown 3DResNet asset-id {asset_id!r}; expected {asset.reference!r}.")
    path = Path(model_path).expanduser() if model_path is not None else SAMPLE_DIR / "model" / asset.filename
    return ModelSelection(asset, resolved, path, model_path is not None)


def build_parser() -> argparse.ArgumentParser:
    """Build the R3D-18 command-line parser with source defaults.

    Returns:
        argparse.ArgumentParser: Parser for selection, input, scheduling,
        and model-free listing/dry-run options.
    """
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
    """Parse the stable CLI contract without reading board identity.

    Args:
        argv: Optional command-line argument sequence, excluding the program
            name. None reads sys.argv through argparse.

    Returns:
        argparse.Namespace: Parsed options.
    """
    return build_parser().parse_args(argv)


def run_list_models(target) -> int:
    """Print the exact manifest asset references for ``target`` (model-free).

    Args:
        target: Concrete target or ``auto``.

    Returns:
        int: 0 after printing the list.
    """
    assets = list_available_assets(target)
    for asset in assets:
        print(asset.reference)
    print(f"{len(assets)} exact asset; no model loaded.")
    return 0


def run_dry_run(selection) -> int:
    """Print the resolved selection contract without loading SDK or files.

    Args:
        selection: Resolved selection to preview.

    Returns:
        int: 0 after printing the preview.
    """
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


def load_labels(path: str | Path) -> dict[int, str]:
    """Load and validate the exact published Kinetics-400 class mapping.

    Args:
        path: JSON file mapping class names to integer class ids.

    Returns:
        dict[int, str]: Class id to class name for all 400 classes.

    Raises:
        ValueError: The file is not a 400-entry mapping of unique integer
            ids covering every id from 0 through 399; malformed JSON raises
            ``json.JSONDecodeError``, a ``ValueError`` subclass.
        OSError: The file cannot be read at all.
    """
    with Path(path).expanduser().open(encoding="utf-8") as handle:
        values = json.load(handle)
    if not isinstance(values, dict) or len(values) != 400:
        raise ValueError("Expected a 400-entry Kinetics class-name mapping.")
    result: dict[int, str] = {}
    for name, class_id in values.items():
        if isinstance(class_id, bool) or not isinstance(class_id, int):
            raise ValueError(f"Invalid Kinetics class id: {class_id!r}; expected JSON integer.")
        index = class_id
        if index in result or not 0 <= index < 400:
            raise ValueError(f"Kinetics class ids must be unique integers in [0,399], got {index}.")
        result[index] = str(name).replace('"', "")
    if set(result) != set(range(400)):
        raise ValueError("Kinetics class ids must cover every id from 0 through 399.")
    return result


def report(result, labels: dict[int, str], selection, clip_path: Path) -> dict:
    """Assemble the JSON report for one finished prediction.

    Args:
        result: ClassificationResult with ranked class_ids and scores.
        labels: Class-id to name mapping; missing ids render as strings.
        selection: Selection describing the executed artifact.
        clip_path: Executed clip path recorded in the report.

    Returns:
        dict: JSON-serializable report payload.
    """
    return {
        "asset_id": selection.asset.reference,
        "target": selection.target,
        "clip": str(clip_path),
        "predictions": [
            {"class_id": int(class_id), "score": float(score), "label": labels.get(int(class_id), str(int(class_id)))}
            for class_id, score in zip(result.class_ids, result.scores)
        ],
    }
