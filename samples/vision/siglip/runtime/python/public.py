# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Pinned Google base224 image encoder, retaining pooled and patch features."""
from pathlib import Path
import argparse
import json
import numpy as np
from utils.py_utils.converted_featuremap import Runner, select
from .public_preprocess import prepare_image

SAMPLE = Path(__file__).resolve().parents[2]
VARIANT = 'base-patch16-224-bicubic-dual-r02'


def validate_outputs(outputs):
    expected = {'pooler_output': (1, 768), 'last_hidden_state': (1, 196, 768)}
    if set(outputs) != set(expected):
        raise ValueError('Both the trained pooler and all patch features are required')
    for name, shape in expected.items():
        value = outputs[name]
        if (not isinstance(value, np.ndarray) or value.shape != shape
                or value.dtype != np.float32 or not np.isfinite(value).all()):
            raise ValueError('Expected complete finite F32 SigLIP outputs')
    return outputs


def patch_projection(outputs):
    """Project actual patch features to three PCA colors; not class labels."""
    patches = validate_outputs(outputs)['last_hidden_state'][0].astype(np.float64)
    centered = patches - patches.mean(axis=0)
    _, _, directions = np.linalg.svd(centered, full_matrices=False)
    directions = directions[:3].copy()
    for direction in directions:
        if direction[np.argmax(np.abs(direction))] < 0:
            direction *= -1
    values = centered @ directions.T
    lo, hi = np.percentile(values, [2, 98], axis=0)
    rgb = np.uint8(np.rint(np.clip((values-lo)/np.maximum(hi-lo, 1e-12), 0, 1)*255))
    return rgb.reshape(14, 14, 3), {
        'method': 'Patch-feature PCA, sign anchored, component percentiles 2–98',
        'semantic_labels': False, 'grid': [14, 14]}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--target', choices=['s100'], default='s100')
    p.add_argument('--variant', default=VARIANT)
    p.add_argument('--asset-id'); p.add_argument('--model-path', type=Path)
    p.add_argument('--image', type=Path)
    p.add_argument('--backend', choices=['board', 'source-onnx'], default='board')
    p.add_argument('--source-onnx', type=Path)
    p.add_argument('--output', type=Path, default=Path('outputs/siglip-public/features.npz'))
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args(argv)
    selection = select(SAMPLE, a.target, a.variant, asset_id=a.asset_id, model_path=a.model_path)
    if a.dry_run:
        print(json.dumps({'asset_id': selection.asset.reference, 'profile': selection.profile})); return 0
    if a.image is None: p.error('--image is required')
    outputs = validate_outputs(Runner(selection, backend=a.backend, source_onnx=a.source_onnx)(prepare_image(a.image)))
    a.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(a.output, **outputs)
    print(json.dumps({'backend': a.backend, 'asset_id': selection.asset.reference,
                      'output': str(a.output), 'shapes': {k: list(v.shape) for k, v in outputs.items()}}))
    return 0


if __name__ == '__main__': raise SystemExit(main())
