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
from pathlib import Path
import sys

from samples.vision.siglip.runtime.python.model_binding import (
    SAMPLE_DIR, SUBMODELS, VARIANTS, list_available_assets, resolve_selection,
)


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


__all__ = ['build_parser', 'read_bgr_image', 'run_dry_run', 'run_list_models',
           'save_feature_tensor', 'summarize_result', 'print_summary']
