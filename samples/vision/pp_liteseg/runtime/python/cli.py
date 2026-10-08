# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""PP-LiteSeg CLI options, published-model selection, listing, dry-run, and output.

``main.py`` uses these helpers to parse arguments, preview a selection, read
the input image, and save the rendered/mask/report artifacts. The
segmentation flow itself lives in ``pp_liteseg.py``.
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
    """A PP-LiteSeg selection or metadata contract violation."""


SUPPORTED_TARGETS = ("x5", "s100", "s100p", "s600")
SAMPLE_DIR = Path(__file__).resolve().parents[2]
ASSET_ID = "x5:pp_liteseg:pp_liteseg_stdc1_cityscapes_1024x512_nv12.bin"


@dataclass(frozen=True)
class ModelSelection:
    """One published manifest asset and its selected local path.

    Attributes:
        target: Concrete execution target (``x5`` for PP-LiteSeg).
        asset: Manifest asset record backing the selection.
        model_path: Local compiled model path.
        explicit_model_path: Whether the caller supplied the path explicitly.
    """

    target: str
    asset: Asset
    model_path: Path
    explicit_model_path: bool = False


def _asset() -> Asset:
    rows = list_assets("x5", "pp_liteseg")
    if len(rows) != 1 or rows[0].reference != ASSET_ID:
        raise BindingError(f"Expected one manifest asset {ASSET_ID!r}.")
    return rows[0]


def list_available_assets(target: Optional[str] = None) -> tuple[Asset, ...]:
    """List the published X5 asset without board or SDK access.

    Args:
        target: Concrete target filter; S targets publish nothing.

    Returns:
        tuple[Asset, ...]: Published assets in manifest order.

    Raises:
        BindingError: The target is unknown.
    """
    if target in (None, "auto", "x5"):
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
    """Resolve PP-LiteSeg's exact published asset identity without loading it.

    Args:
        target: ``auto``/``x5``; other targets publish nothing.
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
        key = "x5"
    if key != "x5":
        raise BindingError("PP-LiteSeg is published only for target x5.")
    asset = _asset()
    if asset_id is not None and asset_id != asset.reference:
        raise BindingError(f"Expected asset-id {asset.reference}, got {asset_id!r}.")
    if model_path is not None and asset_id is None:
        raise BindingError("An external model path requires --asset-id x5:pp_liteseg:pp_liteseg_stdc1_cityscapes_1024x512_nv12.bin.")
    return ModelSelection(
        target=key,
        asset=asset,
        model_path=Path(model_path).expanduser() if model_path else SAMPLE_DIR / "model" / asset.filename,
        explicit_model_path=model_path is not None,
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the PP-LiteSeg command-line parser with source defaults.

    Returns:
        argparse.ArgumentParser: Parser for selection, input, scheduling,
        output, and model-free listing/dry-run options.
    """
    p = argparse.ArgumentParser(description='PP-LiteSeg Cityscapes class-map inference')
    p.add_argument('--target', choices=('auto',) + SUPPORTED_TARGETS, default='auto')
    p.add_argument('--asset-id')
    p.add_argument('--model-path', type=Path)
    p.add_argument('--test-img', type=Path, default=SAMPLE_DIR / 'test_data/street.png')
    p.add_argument('--output', type=Path, default=Path('outputs/pp_liteseg/result.jpg'))
    p.add_argument('--mask-save-path', type=Path, default=Path('outputs/pp_liteseg/labels.npy'))
    p.add_argument('--report-path', type=Path, default=Path('outputs/pp_liteseg/result.json'))
    p.add_argument('--alpha', type=float, default=0.55)
    p.add_argument('--input-width', type=int, default=1024, help='Compiled geometry; must remain 1024')
    p.add_argument('--input-height', type=int, default=512, help='Compiled geometry; must remain 512')
    p.add_argument('--priority', type=int, default=None)
    p.add_argument('--bpu-cores', nargs='+', type=int, default=None)
    modes = p.add_mutually_exclusive_group()
    modes.add_argument('--list-models', action='store_true')
    modes.add_argument('--dry-run', action='store_true')
    return p


def run_list_models(target: str) -> int:
    """Print the published PP-LiteSeg asset as JSON without loading the SDK.

    Args:
        target: Concrete target or ``auto`` to list the publication.

    Returns:
        int: 0 after printing the list.
    """
    print(json.dumps([{'asset_id': a.reference, 'url': a.url, 'sha256': a.sha256, 'target': 'x5'}
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
                      'input_shape': [768, 1024], 'mask_shape': [512, 1024],
                      'output_semantics': 'int32 class IDs 0..18; not logits',
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


def save_results(output: Path, mask_save_path: Path, report_path: Path,
                 result, labels, report: dict) -> None:
    """Write the rendered image, class-mask NPY, and JSON report artifacts.

    Args:
        output: Destination rendered-image path.
        mask_save_path: Destination ``.npy`` class-mask path.
        report_path: Destination JSON report path.
        result: Rendered visualization to save as the image artifact.
        labels: Model-resolution int32 class mask to save.
        report: JSON-serializable report mapping.

    Returns:
        None.

    Raises:
        OSError: A destination directory or file cannot be written.
    """
    import cv2
    import numpy as np

    paths = [Path(output).expanduser(), Path(mask_save_path).expanduser(), Path(report_path).expanduser()]
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(paths[0]), result):
        raise OSError(f'Could not save image {paths[0]}')
    np.save(paths[1], labels, allow_pickle=False)
    paths[2].write_text(json.dumps(report, indent=2) + '\n')


def runtime_report(selection: ModelSelection, binding, runtime_version: str,
                   input_path: Path, labels, result_shape, output: Path,
                   mask_save_path: Path) -> dict:
    """Assemble the JSON-serializable PP-LiteSeg runtime report.

    Args:
        selection: Selection describing the executed artifact.
        binding: Validated model binding supplying metadata evidence.
        runtime_version: Board SDK version string.
        input_path: Executed input image path.
        labels: Predicted model-resolution class mask.
        result_shape: Rendered output image shape recorded in the report.
        output: Destination rendered-image path recorded in the report.
        mask_save_path: Destination mask path recorded in the report.

    Returns:
        dict: Report payload with identity, metadata evidence, and results.
    """
    import numpy as np

    ids = np.unique(labels).tolist()
    return {'target': selection.target, 'asset_id': selection.asset.reference,
            'model_path': str(selection.model_path), 'input_path': str(input_path),
            'publisher_sha256': selection.asset.sha256,
            'runtime_version': runtime_version,
            'metadata': metadata_evidence(binding.metadata), 'class_ids': ids,
            'mask_shape': list(labels.shape), 'output_shape': list(result_shape),
            'output': str(output), 'mask_save_path': str(mask_save_path)}


def class_names_for(ids) -> list:
    """Map numeric class IDs to the fixed Cityscapes class names.

    Args:
        ids: Iterable of integer class IDs present in a mask.

    Returns:
        list: Display names in the same order as ``ids``.
    """
    return [CITYSCAPES_CLASS_NAMES[i] for i in ids]


CITYSCAPES_PALETTE_BGR = (
    (128,  64, 128),  # 0  road
    (232,  35, 244),  # 1  sidewalk
    ( 70,  70,  70),  # 2  building
    (156, 102, 102),  # 3  wall
    (153, 153, 190),  # 4  fence
    (153, 153, 153),  # 5  pole
    ( 30, 170, 250),  # 6  traffic light
    (  0, 220, 220),  # 7  traffic sign
    ( 35, 142, 107),  # 8  vegetation
    (152, 251, 152),  # 9  terrain
    (180, 130,  70),  # 10 sky
    ( 60,  20, 220),  # 11 person
    (  0,   0, 255),  # 12 rider
    (142,   0,   0),  # 13 car
    (100,  60,   0),  # 14 truck
    ( 70,   0,   0),  # 15 bus
    (100,  80,   0),  # 16 train
    (230,   0,   0),  # 17 motorcycle
    ( 32,  11, 119),  # 18 bicycle
)


CITYSCAPES_CLASS_NAMES = [
    "road", "sidewalk", "building", "wall", "fence", "pole",
    "traffic light", "traffic sign", "vegetation", "terrain", "sky",
    "person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle",
]


def colorize(seg: np.ndarray) -> np.ndarray:
    """Map class indices to BGR colors using the Cityscapes palette."""

    import numpy as np

    h, w = seg.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for cid in range(len(CITYSCAPES_PALETTE_BGR)):
        out[seg == cid] = CITYSCAPES_PALETTE_BGR[cid]
    return out


def draw_legend(canvas: np.ndarray, cls_ids: list) -> np.ndarray:
    """Overlay a small class legend on the top-right corner of canvas."""

    import cv2
    import numpy as np

    box, pad = 18, 6
    font, fs, th = cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1
    leg_h = (box + pad) * len(cls_ids) + pad
    leg_w = 155
    leg = np.full((leg_h, leg_w, 3), 30, dtype=np.uint8)
    for i, cid in enumerate(cls_ids):
        y0 = pad + i * (box + pad)
        c = [int(x) for x in CITYSCAPES_PALETTE_BGR[cid]]
        cv2.rectangle(leg, (pad, y0), (pad + box, y0 + box), c, -1)
        cv2.putText(leg, CITYSCAPES_CLASS_NAMES[cid], (pad + box + 4, y0 + box - 3),
                    font, fs, (220, 220, 220), th)
    h, w = canvas.shape[:2]
    canvas[4: 4 + leg_h, w - leg_w - 4: w - 4] = leg
    return canvas


def render_result(bgr: np.ndarray, seg: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    """Produce a 3-panel result image: Original | Overlay | Segmentation.

    Args:
        bgr: Original BGR input image (any size).
        seg: Segmentation map (input_height, input_width) int32.

    Returns:
        Concatenated result image (H+header, 3*W+dividers, 3) uint8.
    """

    import cv2
    import numpy as np

    w, h = 1024, 512

    seg_color = colorize(seg)
    orig_rsz = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_LINEAR)
    overlay = cv2.addWeighted(orig_rsz, 1 - alpha, seg_color, alpha, 0)

    unique = sorted(np.unique(seg).tolist())
    valid = [c for c in unique if 0 <= c < len(CITYSCAPES_CLASS_NAMES)]
    overlay = draw_legend(overlay, valid)

    div = np.full((h, 3, 3), 60, dtype=np.uint8)
    panel = np.hstack([orig_rsz, div, overlay, div, seg_color])

    hdr = np.full((36, panel.shape[1], 3), 35, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(hdr, "Original", (10, 25), font, 0.65, (200, 200, 200), 1)
    cv2.putText(hdr, f"Overlay  alpha={alpha:.2f}", (w + 13, 25), font, 0.65, (200, 200, 200), 1)
    cv2.putText(hdr, "Segmentation", (2 * w + 16, 25), font, 0.65, (200, 200, 200), 1)
    return np.vstack([hdr, panel])
