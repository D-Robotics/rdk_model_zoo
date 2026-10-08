# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0

"""LPRNet CLI options, published-model selection, listing, and dry-run.

``main.py`` uses these helpers to parse arguments and preview a selection.
The recognition flow itself lives in ``lprnet.py``.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from utils.py_utils.assets import Asset, list_assets


class BindingError(ValueError):
    """A LPRNet selection or metadata contract violation."""


SUPPORTED_TARGETS = ("x5", "s100", "s100p", "s600")
SAMPLE_DIR = Path(__file__).resolve().parents[2]
ASSET_ID = "x5:lprnet:lpr.bin"

DEFAULT_TEST_BIN = SAMPLE_DIR / "test_data" / "test_input.dat"


@dataclass(frozen=True)
class ModelSelection:
    """One exact manifest asset and the path selected for execution.

    Attributes:
        target: Concrete execution target (``x5`` for LPRNet).
        asset: Manifest asset record backing the selection.
        model_path: Local compiled model path.
        explicit_model_path: Whether the caller supplied the path explicitly.
    """

    target: str
    asset: Asset
    model_path: Path
    explicit_model_path: bool = False


def _asset() -> Asset:
    rows = list_assets("x5", "lprnet")
    if len(rows) != 1 or rows[0].reference != ASSET_ID:
        raise BindingError(f"Expected one manifest asset {ASSET_ID!r}.")
    return rows[0]


def list_available_assets(target: str | None = None) -> tuple[Asset, ...]:
    """List the one X5 asset without detecting a board or loading an SDK."""

    if target in (None, "auto", "x5"):
        return (_asset(),)
    if target in SUPPORTED_TARGETS:
        return ()
    raise BindingError(f"Unknown target {target!r}.")


def resolve_selection(
    target: str = "auto",
    *,
    asset_id: str | None = None,
    model_path: str | Path | None = None,
) -> ModelSelection:
    """Resolve an exact published LPRNet model without touching the runtime."""

    key = (target or "auto").lower()
    if key == "auto":
        key = "x5"
    if key != "x5":
        raise BindingError("LPRNet is published only for target x5.")
    asset = _asset()
    if asset_id is not None and asset_id != asset.reference:
        raise BindingError(f"Expected asset-id {asset.reference}, got {asset_id!r}.")
    if model_path is not None and asset_id is None:
        raise BindingError("An external model path requires --asset-id x5:lprnet:lpr.bin.")
    return ModelSelection(
        target=key,
        asset=asset,
        model_path=Path(model_path).expanduser() if model_path else SAMPLE_DIR / "model" / asset.filename,
        explicit_model_path=model_path is not None,
    )




def build_parser() -> argparse.ArgumentParser:
    """Build the host-safe LPRNet CLI parser.

    Returns:
        argparse.ArgumentParser: Parser for selection, input, scheduling, and
        model-free listing/dry-run options.
    """
    parser = argparse.ArgumentParser(description="LPRNet license-plate recognition")
    parser.add_argument("--target", choices=("auto",) + SUPPORTED_TARGETS, default="auto")
    parser.add_argument("--asset-id", help="Exact manifest reference, for example x5:lprnet:lpr.bin")
    parser.add_argument("--model-path", help="Existing model path; requires the exact --asset-id")
    parser.add_argument("--test-bin", default=str(DEFAULT_TEST_BIN), help="Packed float32 input .dat path")
    parser.add_argument("--priority", type=int, default=5)
    parser.add_argument("--bpu-cores", nargs="+", type=int, default=[0])
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--list-models", action="store_true")
    modes.add_argument("--dry-run", action="store_true")
    return parser


def validate_scheduling(args: argparse.Namespace) -> None:
    """Reject scheduling values the run would refuse, including during dry-run.

    Args:
        args: Parsed namespace containing priority and bpu_cores.

    Returns:
        None.

    Raises:
        ValueError: Priority or a core index is out of range.
    """
    if type(args.priority) is not int or not 0 <= args.priority <= 255:
        raise ValueError("priority must be an integer between 0 and 255")
    if not args.bpu_cores or any(type(core) is not int or core < 0 for core in args.bpu_cores):
        raise ValueError("bpu-cores must be a non-empty list of non-negative integer indexes")


def run_list_models(target: str) -> int:
    """Print the published LPRNet assets as JSON without loading the SDK.

    Args:
        target: Concrete target or ``auto`` to list the publication.

    Returns:
        int: 0 after printing the list.
    """
    rows = list_available_assets(target)
    print(json.dumps([
        {"asset_id": row.reference, "filename": row.filename, "format": row.format,
         "url": row.url, "sha256": row.sha256, "target": "x5"}
        for row in rows
    ], ensure_ascii=False, indent=2))
    return 0


def run_dry_run(selection) -> int:
    """Print the resolved selection contract without loading SDK or files.

    Args:
        selection: Resolved selection to preview.

    Returns:
        int: 0 after printing the preview.
    """
    print(json.dumps({
        "target": selection.target, "asset_id": selection.asset.reference,
        "model_path": str(selection.model_path), "model_format": selection.asset.format,
        "input_shape": [1, 3, 24, 94], "input_dtype": "float32",
        "output_shape": [1, 68, 18, 1], "output_dtype": "float32",
        "output_layout": ("released lpr.bin binds native logits (1, 68, 18, 1) "
                          "(measured board protocol); the 3D (1, 68, 18) layout "
                          "is the old host/API compatibility contract and no "
                          "published SDK artifact has been observed with it; "
                          "post_process drops only singleton axes to the CTC "
                          "payload (68, 18)"),
        "source_input": "prepacked float32 .dat; no image preprocessing",
        "model_path_exists": selection.model_path.is_file(),
        "sdk_loaded": False, "downloaded": False,
    }, ensure_ascii=False, indent=2))
    return 0
