# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""UNet CLI options, published-model selection, listing, dry-run, and output.

``main.py`` uses these helpers to parse arguments, preview a selection, read
the input image, and save the mask/overlay/report artifacts. The segmentation
flow itself lives in ``unet.py``.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from utils.py_utils.assets import Asset, list_assets
from utils.py_utils.runtime_meta import metadata_evidence


class BindingError(ValueError):
    """A UNet selection or metadata contract violation."""


SUPPORTED_TARGETS = ("x5", "s100", "s100p", "s600")
SAMPLE_DIR = Path(__file__).resolve().parents[2]
VARIANTS = ("resnet18", "resnet34", "resnet50", "resnet101", "resnet152")


@dataclass(frozen=True)
class ModelSelection:
    """One published manifest asset and its selected local path.

    Attributes:
        target: Concrete execution target (``x5`` for UNet).
        asset: Manifest asset record backing the selection.
        model_path: Local compiled model path.
        explicit_model_path: Whether the caller supplied the path explicitly.
        variant: UNet backbone variant name.
    """

    target: str
    asset: Asset
    model_path: Path
    explicit_model_path: bool = False
    variant: str = "resnet18"


def list_available_assets(target: Optional[str] = None) -> tuple[Asset, ...]:
    """List published UNet assets for a target.

    Args:
        target: Concrete target or ``auto``/None for the X5 publication.

    Returns:
        tuple[Asset, ...]: Published assets in manifest order.

    Raises:
        BindingError: The target is unknown.
    """
    if target in (None, "auto", "x5"):
        return tuple(list_assets("x5", "unet"))
    if target in SUPPORTED_TARGETS:
        return ()
    raise BindingError(f"Unknown target {target!r}")


def resolve_selection(target: str = "auto", *, variant: Optional[str] = None,
                      asset_id: Optional[str] = None,
                      model_path: Optional[str] = None) -> ModelSelection:
    """Resolve the published UNet artifact for a command.

    Args:
        target: ``auto``/``x5``; other targets publish nothing.
        variant: Backbone variant; defaults to resnet18.
        asset_id: Qualified manifest reference; a ``model_path`` override
            requires the exact reference.
        model_path: Optional explicit local path for the selected asset.

    Returns:
        ModelSelection: Concrete target, manifest asset, and local path.

    Raises:
        BindingError: The target, variant, asset, or path combination is
            invalid.
    """
    key = (target or "auto").lower()
    if key == "auto":
        key = "x5"
    if key != "x5":
        raise BindingError("UNet has published assets only for x5.")
    rows = list_available_assets(key)
    if asset_id is not None:
        matches = [a for a in rows if a.reference == asset_id]
        if len(matches) != 1:
            raise BindingError(f"Unknown UNet asset {asset_id!r}")
        inferred = matches[0].filename.removeprefix("unet_").split("_voc_")[0]
        if variant is not None and variant != inferred:
            raise BindingError("variant and asset-id select different UNet artifacts")
        variant = inferred
    variant = variant or "resnet18"
    if variant not in VARIANTS:
        raise BindingError(f"Unknown UNet backbone {variant!r}")
    filename = f"unet_{variant}_voc_512x512_nv12.bin"
    matches = [a for a in rows if a.filename == filename]
    if len(matches) != 1:
        raise BindingError(f"Expected one manifest asset for {variant}")
    if model_path is not None and asset_id is None:
        raise BindingError("External model paths require the exact --asset-id")
    return ModelSelection(key, matches[0],
                          Path(model_path).expanduser() if model_path else SAMPLE_DIR / "model" / filename,
                          model_path is not None, variant)


def build_parser() -> argparse.ArgumentParser:
    """Build the UNet command-line parser with source defaults.

    Returns:
        argparse.ArgumentParser: Parser for selection, input, scheduling,
        output, and model-free listing/dry-run options.
    """
    p = argparse.ArgumentParser(description='UNet Pascal VOC semantic segmentation')
    p.add_argument('--target', choices=('auto',) + SUPPORTED_TARGETS, default='auto')
    p.add_argument('--variant', choices=VARIANTS, default=None)
    p.add_argument('--asset-id')
    p.add_argument('--model-path', type=Path)
    p.add_argument('--test-img', type=Path, default=SAMPLE_DIR / 'test_data/2007_000033.jpg')
    p.add_argument('--mask-save-path', type=Path, default=Path('unet_mask.png'))
    p.add_argument('--img-save-path', type=Path, default=Path('unet_result.png'))
    p.add_argument('--report-path', type=Path, default=Path('unet_runtime_report.json'))
    p.add_argument('--priority', type=int)
    p.add_argument('--bpu-core', type=int)
    p.add_argument('--alpha', type=float, default=0.55)
    modes = p.add_mutually_exclusive_group()
    modes.add_argument('--dry-run', action='store_true')
    modes.add_argument('--list-models', action='store_true')
    return p


def run_list_models(target: str) -> int:
    """Print published UNet assets as JSON without loading the SDK.

    Args:
        target: Concrete target or ``auto`` to list the publication.

    Returns:
        int: 0 after printing the list, including when it is empty.
    """
    print(json.dumps([{'asset_id': a.reference, 'filename': a.filename, 'sha256': a.sha256,
                       'url': a.url, 'target': 'x5'} for a in list_available_assets(target)], indent=2))
    return 0


def run_dry_run(selection: ModelSelection) -> int:
    """Print a selection preview as JSON without loading the SDK.

    Args:
        selection: Resolved selection to preview.

    Returns:
        int: 0 after printing the preview.
    """
    print(json.dumps({'target': selection.target, 'variant': selection.variant,
                      'asset_id': selection.asset.reference,
                      'model_path': str(selection.model_path), 'sdk_loaded': False, 'downloaded': False,
                      'model_path_exists': selection.model_path.is_file(),
                      'packed_input_shape': [1, 768, 512, 1],
                      'mask_shape': [512, 512]}, indent=2))
    return 0


def read_image(path: Path) -> np.ndarray:
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
        raise ValueError(f'Could not read image {path}')
    return image


def save_results(mask_save_path: Path, img_save_path: Path, report_path: Path,
                 mask: np.ndarray, overlay: np.ndarray, report: dict) -> None:
    """Write the mask, overlay, and JSON runtime report artifacts.

    Args:
        mask_save_path: Destination class-mask PNG path.
        img_save_path: Destination overlay PNG path.
        report_path: Destination JSON report path.
        mask: uint8 class mask to save as the mask artifact.
        overlay: uint8 visualization to save as the image artifact.
        report: JSON-serializable report mapping.

    Returns:
        None.

    Raises:
        OSError: A destination directory or file cannot be written.
    """
    import cv2

    for path, data in [(mask_save_path, mask), (img_save_path, overlay)]:
        path = path.expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(path), data):
            raise OSError(f'Could not save {path}')
    report_path = report_path.expanduser()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + '\n')


def runtime_report(selection: ModelSelection, binding, runtime_version: str,
                   image_path: Path, mask: np.ndarray, elapsed_ms: float,
                   mask_save_path: Path, img_save_path: Path) -> dict:
    """Assemble the JSON-serializable UNet runtime report.

    Args:
        selection: Selection describing the executed artifact.
        binding: Validated model binding supplying metadata evidence.
        runtime_version: Board SDK version string.
        image_path: Executed input image path.
        mask: Predicted class mask.
        elapsed_ms: Wall-clock predict duration in milliseconds.
        mask_save_path: Destination mask path recorded in the report.
        img_save_path: Destination overlay path recorded in the report.

    Returns:
        dict: Report payload with identity, metadata evidence, and results.
    """
    import numpy as np

    return {'target': selection.target, 'variant': selection.variant,
            'asset_id': selection.asset.reference,
            'model_path': str(selection.model_path), 'image_path': str(image_path),
            'runtime_version': runtime_version,
            'metadata': metadata_evidence(binding.metadata), 'mask_shape': list(mask.shape),
            'classes_present': np.unique(mask).tolist(), 'elapsed_ms': elapsed_ms,
            'mask_save_path': str(mask_save_path), 'img_save_path': str(img_save_path)}


def voc_palette(num_classes: int = 21) -> np.ndarray:
    """Build the deterministic Pascal VOC color palette.

    Args:
        num_classes: Number of palette entries to generate.

    Returns:
        RGB uint8 palette with shape ``[num_classes, 3]``.
    """

    import numpy as np

    palette = np.zeros((num_classes, 3), dtype=np.uint8)
    for class_id in range(num_classes):
        value = class_id
        bit = 0
        while value:
            palette[class_id, 0] |= ((value >> 0) & 1) << (7 - bit)
            palette[class_id, 1] |= ((value >> 1) & 1) << (7 - bit)
            palette[class_id, 2] |= ((value >> 2) & 1) << (7 - bit)
            value >>= 3
            bit += 1
    return palette


def colorize_mask(mask: np.ndarray, num_classes: int = 21) -> np.ndarray:
    """Convert a class-index mask into an OpenCV BGR visualization.

    Args:
        mask: Two-dimensional semantic class-index mask.
        num_classes: Number of valid class identifiers.

    Returns:
        BGR uint8 visualization with shape ``[H, W, 3]``.

    Raises:
        ValueError: If the mask shape or class range is invalid.
    """

    import numpy as np

    if mask.ndim != 2:
        raise ValueError("mask must be two-dimensional")
    if mask.size and (int(mask.min()) < 0 or int(mask.max()) >= num_classes):
        raise ValueError("mask contains an invalid class identifier")
    rgb = voc_palette(num_classes)[mask.astype(np.int64)]
    return np.ascontiguousarray(rgb[..., ::-1])
