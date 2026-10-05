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

from samples.vision.clip.runtime.python.model_binding import (
    SAMPLE_DIR,
    list_available_assets,
    resolve_selection,
)


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
    from samples.vision.clip.runtime.python.visualization import draw_scores, save_image
    save_image(path, draw_scores(image, prompts, result))


__all__ = ['build_parser', 'parse_prompts', 'print_match_result', 'read_bgr_image',
           'run_dry_run', 'run_list_models', 'save_annotated_image']
