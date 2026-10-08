# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Command-line surface for the SigLIP sample.

Option declarations, the model-free listing/dry-run modes, image reading,
result summaries and the optional NumPy save live here so ``main.py`` can
stay a thin, readable entry: parse arguments, construct the task, call
``predict``, show the result.  Nothing in this module extracts features or
loads a board SDK.
"""
from __future__ import annotations
import argparse
import json
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Optional

from utils.py_utils.assets import Asset, list_assets
from utils.py_utils.platforms import resolve_target

SUBMODELS = ('pooler_output', 'last_hidden_state')
SUPPORTED_TARGETS = ('s100', 's100p')
DEFAULT_VARIANT = 'base-patch16-224'
SAMPLE_DIR = Path(__file__).resolve().parents[2]
# size, embedding dimension, patch tokens: literal source evaluator facts.
VARIANTS = {
    'base-patch16-224': (224, 768, 196),
    'base-patch16-384': (384, 768, 576),
    'base-patch16-512': (512, 768, 1024),
    'large-patch16-256': (256, 1024, 256),
    'large-patch16-384': (384, 1024, 576),
    'so400m-patch14-224': (224, 1152, 256),
    'so400m-patch14-384': (384, 1152, 729),
    'so400m-patch16-256-i18n': (256, 1152, 256),
}


@dataclass(frozen=True)
class ModelSelection:
    """Exact asset, actual execution target and selected packed submodel.

    Attributes:
        asset: Manifest asset record backing the selection.
        target: Concrete execution target (s100 or s100p).
        variant: Published variant name.
        model_path: Local compiled model path.
        submodel: Selected packed submodel name.
        image_size: Fixed square input size of the selected variant.
    """

    asset: Asset
    target: str
    variant: str
    model_path: Path
    submodel: str
    image_size: int


def list_available_assets(target: Optional[str] = None) -> tuple[Asset, ...]:
    """Read eight unique manifest assets; both supported targets use these files.

    Args:
        target: Concrete target filter; other known targets publish nothing.

    Returns:
        tuple[Asset, ...]: Published assets in manifest order.

    Raises:
        ValueError: The target is unknown or the publication changed.
    """
    if target not in (None, 'auto', *SUPPORTED_TARGETS):
        if target not in ('x5', 's600'):
            raise ValueError(f'Unknown target: {target}')
        return ()
    assets = list_assets('s', 'siglip')
    expected = {f's100/bpu-siglip-{v}.hbm' for v in VARIANTS}
    if {a.filename for a in assets} != expected or any(a.format != 'hbm' for a in assets):
        raise ValueError('SigLIP publication changed; review its finite contracts first.')
    return assets


def resolve_selection(target: str = 'auto', *, variant: Optional[str] = None,
                      asset_id: Optional[str] = None, model_path: Optional[str] = None,
                      submodel: str = 'pooler_output', image_size: Optional[int] = None,
                      soc_name: Optional[str] = None, board_type: Optional[str] = None) -> ModelSelection:
    """Resolve source-backed S100/S100P support, never infer it from file names.

    Args:
        target: ``auto`` resolves the executing board; s100/s100p are the
            published targets.
        variant: Published variant; defaults to base-patch16-224.
        asset_id: Qualified manifest reference; a ``model_path`` override
            requires the exact reference.
        model_path: Optional explicit local path for the selected asset.
        submodel: Packed submodel to execute (default pooler_output).
        image_size: Optional assertion; must equal the variant's fixed size.
        soc_name: Optional board-identity override for ``auto`` resolution.
        board_type: Optional board-type override for ``auto`` resolution.

    Returns:
        ModelSelection: Concrete target, variant, submodel, and local path.

    Raises:
        ValueError: The target, variant, asset, or path combination is
            invalid.
    """
    target = resolve_target(target, soc_name=soc_name, board_type=board_type)
    if target not in SUPPORTED_TARGETS:
        raise ValueError(f'No published SigLIP support for {target}; use s100/s100p.')
    if submodel not in SUBMODELS:
        raise ValueError(f'Unknown SigLIP submodel: {submodel}')
    if model_path is not None and asset_id is None:
        raise ValueError('An external model-path requires the exact manifest asset-id.')
    assets = list_available_assets(target)
    if asset_id is None:
        variant = DEFAULT_VARIANT if variant is None else variant
        asset_id = f's:siglip:s100/bpu-siglip-{variant}.hbm'
    matches = [a for a in assets if a.reference == asset_id]
    if len(matches) != 1:
        raise ValueError(f'Unknown SigLIP asset-id: {asset_id}')
    asset = matches[0]
    actual = next(v for v in VARIANTS if asset.filename == f's100/bpu-siglip-{v}.hbm')
    if variant is not None and variant != actual:
        raise ValueError(f'Variant {variant!r} does not match asset {asset.reference}.')
    size = VARIANTS[actual][0]
    if image_size is not None and image_size != size:
        raise ValueError(f'image-size must be {size} for {actual}, got {image_size}.')
    path = Path(model_path).expanduser() if model_path is not None else SAMPLE_DIR / 'model' / asset.filename
    return ModelSelection(asset, target, actual, path, submodel, size)


def build_parser() -> argparse.ArgumentParser:
    """Build the SDK-free parser; no detection, model loading or downloads."""
    p = argparse.ArgumentParser(description='SigLIP packed vision encoder on S100/S100P.')
    p.add_argument('--target', choices=('auto','x5','s100','s100p','s600'), default='auto',
                   help='Actual execution target; auto detects the board.')
    p.add_argument('--variant', choices=tuple(VARIANTS), default=None,
                   help='Published variant; defaults to base-patch16-224 unless asset-id selects one.')
    p.add_argument('--asset-id', default=None, help='Exact qualified manifest reference; required with model-path.')
    p.add_argument('--model-path', default=None, help='Explicit local HBM path, paired with asset-id.')
    p.add_argument('--test-img', default=str(SAMPLE_DIR/'test_data/dog.jpg'),
                   help='Input BGR image; default bundled dog.jpg.')
    p.add_argument('--image-size', type=int, default=None,
                   help='Optional assertion of the selected artifact size; defaults to its bound size.')
    p.add_argument('--submodel', choices=SUBMODELS, default='pooler_output', help='Packed submodel to execute.')
    p.add_argument('--priority', type=int, default=0, help='Runtime priority 0..255.')
    p.add_argument('--bpu-cores', type=int, nargs='+', default=[0], help='Nonnegative BPU core indexes.')
    p.add_argument('--output-file', default=None, help='Optional NumPy-format result path (used exactly as given).')
    mode=p.add_mutually_exclusive_group()
    mode.add_argument('--list-models', action='store_true', help='List unique manifest assets without hardware.')
    mode.add_argument('--dry-run', action='store_true',
                      help='Resolve metadata contract; requires an explicit target, no SDK.')
    return p


def run_list_models(target: str) -> int:
    """Print the unique manifest asset references for ``target`` (model-free)."""
    assets = list_available_assets(target)
    for asset in assets:
        print(asset.reference)
    print(f'{len(assets)} unique assets; supported targets: s100, s100p; no model loaded.')
    return 0


def run_dry_run(selection) -> int:
    """Print the resolved static contract without board, SDK or image loading."""
    size, dim, count = VARIANTS[selection.variant]
    print(json.dumps(dict(target=selection.target, variant=selection.variant,
        asset_id=selection.asset.reference, model_path=str(selection.model_path),
        model_path_exists=selection.model_path.is_file(), submodel=selection.submodel,
        input_shape=[1,3,size,size],
        output_shapes=[[1,dim],[1,1,dim]] if selection.submodel=='pooler_output' else [[1,count,dim]],
        output_policy='preserve native numeric dtype/shape; no dequantization or activation',
        source_manifest=selection.asset.source_path), indent=2))
    return 0


def read_bgr_image(path: str):
    """Read one BGR image; failures name the exact path."""
    import cv2
    image = cv2.imread(str(Path(path).expanduser()))
    if image is None:
        raise ValueError(f'Cannot read image: {path}')
    return image


def summarize_result(result, submodel: str) -> dict:
    """Summarize one owned feature tensor (presentation only)."""
    import numpy as np
    flat = result.reshape(-1).astype(np.float32)
    return dict(submodel=submodel, shape=list(result.shape), dtype=str(result.dtype),
                mean=float(flat.mean()), std=float(flat.std()), min=float(flat.min()),
                max=float(flat.max()), l2_norm=float((flat**2).sum()**0.5))


def print_summary(summary: dict) -> None:
    """Print the JSON feature summary for one finished prediction."""
    print(json.dumps(summary, indent=2, allow_nan=False))


def save_feature_tensor(path: str, result) -> None:
    """Write the owned feature tensor to exactly the requested NumPy path."""
    import numpy as np
    output = Path(path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('wb') as handle:
        np.save(handle, result, allow_pickle=False)
    print(f'Feature tensor saved: {output}')


__all__ = ['DEFAULT_VARIANT', 'ModelSelection', 'SUBMODELS', 'SUPPORTED_TARGETS',
           'VARIANTS', 'build_parser', 'list_available_assets', 'read_bgr_image',
           'resolve_selection', 'run_dry_run', 'run_list_models',
           'save_feature_tensor', 'summarize_result', 'print_summary']
