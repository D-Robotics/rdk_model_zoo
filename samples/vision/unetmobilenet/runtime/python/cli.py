# Copyright (c) 2025-2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""UnetMobileNet CLI options, published-model selection, listing, dry-run, and output.

``main.py`` uses these helpers to parse arguments, preview a selection, read
the input image, and save the overlay/mask/report artifacts. The segmentation
flow itself lives in ``unetmobilenet.py``.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from utils.py_utils.assets import Asset, list_assets
from utils.py_utils.platforms import resolve_target
from utils.py_utils.runtime_meta import metadata_evidence

SAMPLE_DIR = Path(__file__).resolve().parents[2]
SUPPORTED_TARGETS = ('x5', 's100', 's100p', 's600')
FILE_NAME = 'unet_mobilenet_1024x2048_nv12.hbm'


@dataclass(frozen=True)
class ModelSelection:
    """One published manifest asset and its selected local path.

    Attributes:
        target: Concrete execution target (s100 or s600).
        asset: Manifest asset record backing the selection.
        model_path: Local compiled model path.
        explicit_model_path: Whether the caller supplied the path explicitly.
    """

    target: str
    asset: Asset
    model_path: Path
    explicit_model_path: bool = False


def list_available_assets(target: Optional[str] = None) -> tuple[Asset, ...]:
    """List published UnetMobileNet assets for a target.

    Args:
        target: Concrete target filter; ``auto``/None lists all publications.

    Returns:
        tuple[Asset, ...]: Published assets in manifest order.

    Raises:
        ValueError: The target is unknown.
    """
    rows = tuple(list_assets('s', 'unetmobilenet'))
    if target in (None, 'auto'):
        return rows
    if target not in SUPPORTED_TARGETS:
        raise ValueError(f'Unknown target {target!r}')
    return tuple(a for a in rows if a.filename.startswith(target + '/'))


def resolve_selection(target: str = 'auto', *, asset_id: Optional[str] = None,
                      model_path: Optional[str] = None) -> ModelSelection:
    """Resolve the published UnetMobileNet artifact for a command.

    Args:
        target: ``auto``/None resolves the executing board; s100 or s600 are
            the published targets.
        asset_id: Qualified manifest reference; a ``model_path`` override
            requires the exact reference.
        model_path: Optional explicit local path for the selected asset.

    Returns:
        ModelSelection: Concrete target, manifest asset, and local path.

    Raises:
        ValueError: The target, asset, or path combination is invalid.
    """
    rows = list_available_assets()
    if asset_id is not None:
        matches = [asset for asset in rows if asset.reference == asset_id]
        if len(matches) != 1:
            raise ValueError(f'Unknown UnetMobileNet asset-id {asset_id!r}')
        inferred = matches[0].filename.split('/')[0]
        if target in (None, 'auto'):
            target = inferred
        elif target != inferred:
            raise ValueError('target and asset-id select different artifacts')
    key = resolve_target(target) if target in (None, 'auto') else target
    if key not in ('s100', 's600'):
        raise ValueError(f'UnetMobileNet has no published asset for {key}')
    matches = [asset for asset in rows if asset.filename == f'{key}/{FILE_NAME}']
    if len(matches) != 1:
        raise ValueError(f'Expected one UnetMobileNet asset for {key}')
    if model_path is not None and asset_id is None:
        raise ValueError('External model paths require the exact --asset-id')
    asset = matches[0]
    path = Path(model_path).expanduser() if model_path is not None else SAMPLE_DIR / 'model' / asset.filename
    return ModelSelection(key, asset, path, model_path is not None)


def build_parser() -> argparse.ArgumentParser:
    """Build the UnetMobileNet command-line parser with source defaults.

    Returns:
        argparse.ArgumentParser: Parser for selection, input, scheduling,
        output, and model-free listing/dry-run options.
    """
    parser = argparse.ArgumentParser(
        description='UnetMobileNet Cityscapes semantic segmentation (split NV12)')
    parser.add_argument('--target', choices=('auto',) + SUPPORTED_TARGETS, default='auto')
    parser.add_argument('--asset-id')
    parser.add_argument('--model-path', type=Path)
    parser.add_argument('--test-img', type=Path, default=SAMPLE_DIR / 'test_data/segmentation.png')
    parser.add_argument('--img-save-path', type=Path, default=Path('result.jpg'))
    parser.add_argument('--mask-save-path', type=Path, default=Path('unetmobilenet_mask.npy'))
    parser.add_argument('--report-path', type=Path, default=Path('unetmobilenet_report.json'))
    parser.add_argument('--alpha-f', type=float, default=0.75, help='Weight of ORIGINAL image, 0..1')
    parser.add_argument('--priority', type=int, default=0)
    parser.add_argument('--bpu-cores', nargs='+', type=int, default=[0])
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--list-models', action='store_true')
    mode.add_argument('--dry-run', action='store_true')
    return parser


def run_list_models(target: str) -> int:
    """Print published UnetMobileNet assets as JSON without loading the SDK.

    Args:
        target: Concrete target or ``auto`` to list publications.

    Returns:
        int: 0 after printing the list.
    """
    print(json.dumps([{'asset_id': a.reference, 'url': a.url, 'sha256': a.sha256}
                      for a in list_available_assets(target)], indent=2))
    return 0


def run_dry_run(selection: ModelSelection, test_img: Path) -> int:
    """Print a selection preview as JSON without loading the SDK.

    Args:
        selection: Resolved selection to preview.
        test_img: Input image path recorded in the preview.

    Returns:
        int: 0 after printing the preview.
    """
    print(json.dumps({'target': selection.target, 'asset_id': selection.asset.reference,
                      'model_path': str(selection.model_path), 'test_img': str(test_img),
                      'input_shapes': [[1, 1024, 2048, 1], [1, 512, 1024, 2]],
                      'output': 'original-resolution int32 class IDs 0..18',
                      'sdk_loaded': False, 'downloaded': False}, indent=2))
    return 0


def read_image(path: Path):
    """Read the input image as BGR uint8.

    Args:
        path: Image file path; leading ~ is expanded.

    Returns:
        np.ndarray: uint8 BGR array shaped (H, W, 3).

    Raises:
        ValueError: OpenCV cannot decode the file.
    """
    import cv2

    image = cv2.imread(str(Path(path).expanduser()), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f'Cannot read image: {path}')
    return image


def save_results(img_save_path: Path, mask_save_path: Path, report_path: Path,
                 overlay, labels, report: dict) -> None:
    """Write the overlay image, class-mask NPY, and JSON report artifacts.

    Args:
        img_save_path: Destination overlay image path.
        mask_save_path: Destination ``.npy`` class-mask path.
        report_path: Destination JSON report path.
        overlay: Visualization to save as the image artifact.
        labels: Original-resolution int32 class mask to save.
        report: JSON-serializable report mapping.

    Returns:
        None.

    Raises:
        OSError: A destination directory or file cannot be written.
    """
    import cv2
    import numpy as np

    image_path, mask_path = Path(img_save_path).expanduser(), Path(mask_save_path).expanduser()
    report_path = Path(report_path).expanduser()
    for path in (image_path, mask_path, report_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(image_path), overlay):
        raise OSError(f'Cannot write {image_path}')
    np.save(mask_path, labels, allow_pickle=False)
    report_path.write_text(json.dumps(report, indent=2) + '\n')


def runtime_report(selection: ModelSelection, binding, runtime_version: str,
                   input_path: Path, labels, alpha_f: float,
                   img_save_path: Path, mask_save_path: Path) -> dict:
    """Assemble the JSON-serializable UnetMobileNet runtime report.

    Args:
        selection: Selection describing the executed artifact.
        binding: Validated model binding supplying metadata evidence.
        runtime_version: Board SDK version string.
        input_path: Executed input image path.
        labels: Predicted original-resolution class mask.
        alpha_f: Overlay blend weight recorded in the report.
        img_save_path: Destination overlay path recorded in the report.
        mask_save_path: Destination mask path recorded in the report.

    Returns:
        dict: Report payload with identity, metadata evidence, and results.
    """
    import numpy as np

    return {
        'target': selection.target, 'asset_id': selection.asset.reference,
        'model_path': str(selection.model_path), 'input_path': str(input_path),
        'publisher_sha256': selection.asset.sha256,
        'runtime_version': runtime_version,
        'metadata': metadata_evidence(binding.metadata),
        'mask_shape': list(labels.shape), 'class_ids': np.unique(labels).tolist(),
        'alpha_f': alpha_f, 'img_save_path': str(img_save_path), 'mask_save_path': str(mask_save_path),
    }


# Preserved from rdk_s utils/py_utils/visualize.py, consumed as BGR by source.
PALETTE_BGR = (
    (56, 56, 255), (151, 157, 255), (31, 112, 255), (29, 178, 255),
    (49, 210, 207), (10, 249, 72), (23, 204, 146), (134, 219, 61),
    (52, 147, 26), (187, 212, 0), (168, 153, 44), (255, 194, 0),
    (147, 69, 52), (255, 115, 100), (236, 24, 0), (255, 56, 132),
    (133, 0, 82), (255, 56, 203), (200, 149, 255), (199, 55, 255)
)


def render_overlay(image, labels, *, alpha_f=0.75):
    """Blend original BGR with source colors; neither inference nor file IO."""

    import cv2
    import numpy as np

    if (not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3
            or image.dtype != np.uint8 or not all(image.shape[:2])):
        raise ValueError('Expected nonempty BGR uint8 HWC image')
    if (not isinstance(labels, np.ndarray) or labels.shape != image.shape[:2]
            or labels.dtype != np.int32 or np.any(labels < 0) or np.any(labels >= 19)):
        raise ValueError('Expected original-resolution int32 class IDs 0..18')
    if not 0 <= alpha_f <= 1:
        raise ValueError('alpha_f must be in [0,1]')
    return cv2.addWeighted(image, alpha_f, np.asarray(PALETTE_BGR, dtype=np.uint8)[labels], 1-alpha_f, 0.0)
