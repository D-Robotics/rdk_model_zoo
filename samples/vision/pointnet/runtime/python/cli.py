# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""PointNet CLI surface: option declarations, report assembly and file IO.

``main.py`` stays a thin entry that constructs the task and calls ``predict``;
everything presentational — the parser, the model-free listing/dry-run modes
and the label/plot/report writing — lives here.  Nothing in this module runs
inference or plots by itself.
"""
from __future__ import annotations
import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from utils.py_utils.assets import Asset, list_assets


class BindingError(ValueError):
    """A PointNet selection or metadata contract violation."""


SUPPORTED_TARGETS = ("x5", "s100", "s100p", "s600")
SAMPLE_DIR = Path(__file__).resolve().parents[2]
ASSET_ID = "s:pointnet:s100/pointnet.hbm"

#: Part names in the fixed source label order used by the result report.
PART_NAMES = ('back', 'seat', 'leg', 'arm')


@dataclass(frozen=True)
class ModelSelection:
    """One published manifest asset and its selected local path.

    Attributes:
        target: Concrete execution target (``s100`` for PointNet).
        asset: Manifest asset record backing the selection.
        model_path: Local compiled model path.
        explicit_model_path: Whether the caller supplied the path explicitly.
    """

    target: str
    asset: Asset
    model_path: Path
    explicit_model_path: bool = False


def _asset() -> Asset:
    rows = list_assets("s", "pointnet")
    if len(rows) != 1 or rows[0].reference != ASSET_ID:
        raise BindingError(f"Expected one manifest asset {ASSET_ID!r}.")
    return rows[0]


def list_available_assets(target: Optional[str] = None) -> tuple[Asset, ...]:
    """List the published S100 asset without board or SDK access.

    Args:
        target: Concrete target filter; other targets publish nothing.

    Returns:
        tuple[Asset, ...]: Published assets in manifest order.

    Raises:
        BindingError: The target is unknown.
    """
    if target in (None, "auto", "s100"):
        return (_asset(),)
    if target in SUPPORTED_TARGETS:
        return ()
    raise BindingError(f"Unknown target {target!r}.")


def resolve_selection(
    target: str = "auto",
    *,
    asset_id: Optional[str] = None,
    model_path: "str | Path | None" = None,
) -> ModelSelection:
    """Resolve PointNet's exact published asset identity without loading it.

    Args:
        target: ``auto``/``s100``; other targets publish nothing.
        asset_id: Qualified manifest reference; a ``model_path`` override
            requires the exact reference.
        model_path: Optional explicit local path for the selected asset.

    Returns:
        ModelSelection: Concrete target, manifest asset, and local path.

    Raises:
        BindingError: The target, asset, or path combination is invalid.
    """
    key = (target or "auto").lower()
    if key == "auto":
        key = "s100"
    if key != "s100":
        raise BindingError("PointNet is published only for target s100.")
    asset = _asset()
    if asset_id is not None and asset_id != asset.reference:
        raise BindingError(f"Expected asset-id {asset.reference}, got {asset_id!r}.")
    if model_path is not None and asset_id is None:
        raise BindingError("An external model path requires --asset-id s:pointnet:s100/pointnet.hbm.")
    return ModelSelection(
        target=key,
        asset=asset,
        model_path=Path(model_path).expanduser() if model_path else SAMPLE_DIR / "model" / asset.filename,
        explicit_model_path=model_path is not None,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='PointNet chair part segmentation')
    parser.add_argument('--target', choices=('auto',) + SUPPORTED_TARGETS, default='auto')
    parser.add_argument('--asset-id')
    parser.add_argument('--model-path', help='External HBM path; requires exact --asset-id')
    parser.add_argument('--test-pts', type=Path, default=SAMPLE_DIR/'test_data/chair.pts')
    parser.add_argument('--output-dir', type=Path, default=Path('outputs/pointnet'))
    parser.add_argument('--no-plot', action='store_true', help='Save labels/report without matplotlib images')
    parser.add_argument('--priority', type=int, default=0)
    parser.add_argument('--bpu-cores', nargs='+', type=int, default=[0])
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--list-models', action='store_true')
    modes.add_argument('--dry-run', action='store_true')
    return parser


def run_list_models(target: str) -> int:
    """Print the manifest-backed assets for ``target`` (model-free)."""
    print(json.dumps([{'asset_id': a.reference, 'target': 's100', 'url': a.url,
                       'sha256': a.sha256} for a in list_available_assets(target)], indent=2))
    return 0


def run_dry_run(selection) -> int:
    """Print the resolved selection without loading a model or SDK."""
    print(json.dumps({'target': selection.target, 'asset_id': selection.asset.reference,
                      'model_path': str(selection.model_path), 'input': '(1,3,N) float32',
                      'output': '(1,N,4) logits; N and raw dtype checked at load',
                      'sdk_loaded': False, 'downloaded': False,
                      'model_path_exists': selection.model_path.is_file()}, indent=2))
    return 0


def save_pointnet_evidence(out: Path, *, selection, binding, input_path, details,
                           no_plot: bool) -> None:
    """Write labels, optional views and the normalization-context report."""
    import numpy as np
    from utils.py_utils.runtime_meta import metadata_evidence

    labels = details.labels
    prepared = details.prepared
    out.mkdir(parents=True, exist_ok=True)
    np.save(out/'labels.npy', labels, allow_pickle=False)
    if not no_plot:
        from samples.vision.pointnet.runtime.python.visualization import (
            save_original_view, save_segmentation_view,
        )
        normalized = prepared.tensors[binding.input_name][0].T
        save_original_view(normalized, str(out/'result_orig.png'))
        save_segmentation_view(normalized, labels, str(out/'result.png'))
    counts = {name: int(np.count_nonzero(labels == i))
              for i, name in enumerate(PART_NAMES)}
    report = {'target': selection.target, 'asset_id': selection.asset.reference,
              'input': str(input_path), 'point_count': len(labels), 'counts': counts,
              'normalization': {'centroid': prepared.context.centroid,
                                'radius': prepared.context.radius},
              'metadata': metadata_evidence(binding.metadata), 'output_dir': str(out)}
    text = json.dumps(report, indent=2)
    (out / 'result.json').write_text(text + '\n')
    print(text)
