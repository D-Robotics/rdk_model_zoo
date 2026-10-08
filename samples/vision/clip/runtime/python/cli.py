# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Command-line surface for the CLIP sample.

Option declarations, the model-free listing/dry-run modes, prompt parsing,
result presentation and the annotated-image write live here so ``main.py``
can stay a thin, readable entry: parse arguments, construct the task, call
``predict``, show the result.  Nothing in this module matches images with
text or loads a board SDK.
"""
import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

from dataclasses import dataclass
from pathlib import Path

from utils.py_utils.assets import Asset, list_assets
from utils.py_utils.platforms import resolve_target

SAMPLE_DIR = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ModelSelection:
    """One published X5 encoder pair and its selected local paths.

    Attributes:
        target: Concrete execution target (``x5`` for CLIP).
        image_asset: Manifest asset of the BPU image encoder.
        text_asset: Manifest asset of the CPU ONNX text encoder.
        image_model_path: Local compiled image-encoder path.
        text_model_path: Local ONNX text-encoder path.
    """

    target: str
    image_asset: Asset
    text_asset: Asset
    image_model_path: Path
    text_model_path: Path


def list_available_assets(target=None):
    """List the exact pair; other known boards have no published CLIP pair.

    Args:
        target: Concrete target filter; S targets publish nothing.

    Returns:
        tuple[Asset, ...]: Published assets in manifest order.

    Raises:
        ValueError: The target is unknown or the publication changed.
    """
    if target in ('s100', 's100p', 's600'):
        return ()
    if target not in (None, 'auto', 'x5'):
        raise ValueError(f'Unknown target: {target}')
    assets = tuple(list_assets('x5', 'clip'))
    if len(assets) != 2 or {(a.filename, a.format) for a in assets} != {
        ('img_encoder.bin', 'bin'), ('text_encoder.onnx', 'onnx')
    }:
        raise ValueError('CLIP publication changed; review the encoder pair first.')
    return assets


def resolve_selection(target='auto', *, image_asset_id=None, text_asset_id=None,
                      image_model_path=None, text_model_path=None):
    """Select both encoders; external paths require the corresponding exact ID.

    Args:
        target: ``auto`` resolves the executing board; only x5 publishes.
        image_asset_id: Optional exact image-encoder manifest reference.
        text_asset_id: Optional exact text-encoder manifest reference.
        image_model_path: Optional explicit local image-encoder path.
        text_model_path: Optional explicit local text-encoder path.

    Returns:
        ModelSelection: Concrete target and the exact asset pair with paths.

    Raises:
        ValueError: The target or asset/path combination is invalid.
    """
    target = resolve_target(target)
    if target != 'x5':
        raise ValueError(f'No published CLIP encoder pair for {target}.')
    assets = {a.filename: a for a in list_available_assets(target)}
    selected = []
    paths = []
    for filename, asset_id, path in (
        ('img_encoder.bin', image_asset_id, image_model_path),
        ('text_encoder.onnx', text_asset_id, text_model_path),
    ):
        asset = assets[filename]
        if path is not None and asset_id is None:
            raise ValueError(f'External {filename} path requires its exact asset-id.')
        if asset_id is not None and asset_id != asset.reference:
            raise ValueError(f'Expected exact asset-id {asset.reference}, got {asset_id!r}.')
        selected.append(asset)
        paths.append(Path(path).expanduser() if path is not None else SAMPLE_DIR / 'model' / filename)
    return ModelSelection(target, *selected, *paths)


def build_parser():
    """Build parser without importing BPU/ONNX runtimes or initializing BPE."""
    parser = argparse.ArgumentParser(description='CLIP X5 BPU image + CPU ONNX text matching.')
    parser.add_argument('--target', choices=('auto', 'x5', 's100', 's100p', 's600'), default='auto',
                        help='Actual execution target; only X5 has a published pair.')
    parser.add_argument('--image-asset-id', default=None, help='Exact image asset ID; required with image-model-path.')
    parser.add_argument('--text-asset-id', default=None, help='Exact text asset ID; required with text-model-path.')
    parser.add_argument('--image-model-path', default=None, help='Explicit local .bin; default published img_encoder.bin.')
    parser.add_argument('--text-model-path', default=None, help='Explicit local .onnx; default published text_encoder.onnx.')
    parser.add_argument('--test-img', default=str(SAMPLE_DIR / 'test_data/dog.jpg'),
                        help='Input BGR image path.')
    parser.add_argument('--texts', default='a diagram,a dog',
                        help='Comma-separated prompts; empty items are removed.')
    parser.add_argument('--img-save-path', default=str(SAMPLE_DIR / 'test_data/inference.png'),
                        help='Annotated image destination; retains source default.')
    parser.add_argument('--priority', type=int, default=0, help='Image encoder priority 0..255.')
    parser.add_argument('--bpu-cores', type=int, nargs='+', default=[0],
                        help='Nonnegative image encoder BPU cores.')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--list-models', action='store_true',
                      help='List the exact pair without loading either runtime.')
    mode.add_argument('--dry-run', action='store_true',
                      help='Resolve the pair with explicit target, without board or SDK.')
    return parser


def run_list_models(target: str) -> int:
    """Print the published pair references for ``target`` (model-free)."""
    for asset in list_available_assets(target):
        print(asset.reference)
    return 0


def run_dry_run(selection) -> int:
    """Print the resolved static contract without board, SDK or BPE loading."""
    print(json.dumps({'target': selection.target,
                      'image_asset_id': selection.image_asset.reference,
                      'text_asset_id': selection.text_asset.reference,
                      'image_model_path': str(selection.image_model_path),
                      'text_model_path': str(selection.text_model_path),
                      'image_input': 'F32[1,3,224,224] RGB [0,1]',
                      'text_input': 'I32[N,77] BPE',
                      'outputs': 'F32[1,512] and F32[N,512]; cosine, no softmax'}, indent=2))
    return 0


def parse_prompts(texts: str) -> list:
    """Split one comma-separated prompt string; empty items are removed."""
    prompts = [text.strip() for text in texts.split(',') if text.strip()]
    if not prompts:
        raise ValueError('At least one nonempty text prompt is required.')
    return prompts


def read_bgr_image(path: str):
    """Read one BGR image; failures name the exact path."""
    import cv2
    image = cv2.imread(str(Path(path).expanduser()))
    if image is None:
        raise ValueError(f'Cannot read image: {path}')
    return image


def print_match_result(result, selection, prompts: Sequence, image_saved: str) -> None:
    """Print the JSON match report for one finished prediction."""
    print(json.dumps({'target': selection.target, 'prompts': list(prompts),
                      'scores': result.scores.tolist(), 'order': result.order.tolist(),
                      'image_saved': str(Path(image_saved).expanduser())},
                     indent=2, ensure_ascii=False, allow_nan=False))


def save_annotated_image(path: str, image, prompts, result) -> None:
    """Write the annotated match image (presentation only)."""
    save_image(path, draw_scores(image, prompts, result))


__all__ = ['build_parser', 'parse_prompts', 'print_match_result', 'read_bgr_image',
           'run_dry_run', 'run_list_models', 'save_annotated_image', 'draw_scores', 'save_image']


def draw_scores(image, texts, result):
    """Return a copy annotated with each prompt's ranked cosine similarity."""

    import cv2

    canvas = image.copy()
    for rank, index in enumerate(result.order, start=1):
        text = f'Rank {rank}: {texts[int(index)]} | similarity: {result.scores[int(index)]:.4f}'
        cv2.putText(canvas, text, (10, 40+(rank-1)*40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)
    return canvas


def save_image(path, image):
    """Write exactly the caller's visualization path or raise an explicit error."""

    import cv2

    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f'Failed to save image to {path}')
