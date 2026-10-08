# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Command-line surface for the MobileSAM sample.

Option declarations (including the box-prompt parser), the model-free
listing/dry-run modes, image reading and the overlay/mask writes live here
so ``main.py`` can stay a thin, readable entry: parse arguments, construct
the pipeline, call ``predict``, show the result.  Nothing in this module
segments images or loads a board SDK.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from utils.py_utils import sam_binding as _shared

SAMPLE = "mobile_sam"
SAMPLE_DIR = Path(__file__).resolve().parents[2]
SUPPORTED_TARGETS = ("x5", "s100", "s100p", "s600")

ModelSelection = _shared.ModelSelection
ModelBinding = _shared.ModelBinding


def list_available_assets(target=None):
    """Return manifest-backed encoder/decoder assets for the selected target.

    Args:
        target: Concrete target filter; ``auto``/None lists all publications.

    Returns:
        The shared catalog's exact asset rows for this sample.
    """
    return _shared.list_available_assets(SAMPLE, target)


def resolve_selection(
    target="auto",
    *,
    encoder_model_path=None,
    decoder_model_path=None,
    encoder_asset_id=None,
    decoder_asset_id=None,
):
    """Resolve one exact published encoder/decoder pair.

    Args:
        target: ``auto`` resolves the executing board; concrete targets
            filter the published pairs.
        encoder_model_path: Optional explicit local encoder path.
        decoder_model_path: Optional explicit local decoder path.
        encoder_asset_id: Qualified encoder manifest reference.
        decoder_asset_id: Qualified decoder manifest reference.

    Returns:
        ModelSelection: Concrete target and the exact asset pair with paths.

    Raises:
        ValueError: The target or asset combination is invalid.
    """
    return _shared.resolve_selection(
        SAMPLE,
        target,
        encoder_model_path=encoder_model_path,
        decoder_model_path=decoder_model_path,
        encoder_asset_id=encoder_asset_id,
        decoder_asset_id=decoder_asset_id,
        sample_dir=SAMPLE_DIR,
    )

DEFAULT_TEST_IMAGE = SAMPLE_DIR / "test_data" / "dogs.jpg"
DEFAULT_RESULT_IMAGE = SAMPLE_DIR / "test_data" / "mobile_sam_full_mask_result.jpg"
DEFAULT_MASK_IMAGE = SAMPLE_DIR / "test_data" / "mobile_sam_binary_mask_result.png"
DEFAULT_BOX = (185.0, 120.0, 380.0, 445.0)


def parse_box(value: str) -> tuple:
    """Parse one ``x1,y1,x2,y2`` box prompt in resized 512 coordinates."""
    try:
        values = tuple(float(coordinate) for coordinate in value.split(","))
        if len(values) != 4 or not all(math.isfinite(coordinate) for coordinate in values):
            raise ValueError("box must contain four finite coordinates")
        x1, y1, x2, y2 = values
        if not (0.0 <= x1 < x2 <= 512.0 and 0.0 <= y1 < y2 <= 512.0):
            raise ValueError("box must be ordered and lie in [0, 512]")
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("box must be x1,y1,x2,y2") from exc
    return values


def build_parser() -> argparse.ArgumentParser:
    """Build the SDK-free parser; ``main`` returns zero or a user error code."""
    parser = argparse.ArgumentParser(description="MobileSAM dual-model box-prompt mask segmentation.")
    parser.add_argument("--target", choices=("auto", "x5", "s100", "s100p", "s600"), default="auto")
    parser.add_argument("--encoder-asset-id", default=None, help="Exact manifest encoder asset reference.")
    parser.add_argument("--decoder-asset-id", default=None, help="Exact manifest decoder asset reference.")
    parser.add_argument("--encoder-model-path", default=None, help="External encoder model path; requires its asset ID.")
    parser.add_argument("--decoder-model-path", default=None, help="External decoder model path; requires its asset ID.")
    parser.add_argument("--test-img", default=str(DEFAULT_TEST_IMAGE), help="Input BGR image path.")
    parser.add_argument("--img-save-path", default=str(DEFAULT_RESULT_IMAGE), help="Overlay output path.")
    parser.add_argument("--mask-save-path", default=str(DEFAULT_MASK_IMAGE), help="Binary mask output path.")
    parser.add_argument("--box", type=parse_box, default=DEFAULT_BOX,
                        help="Box x1,y1,x2,y2 in resized 512x512 coordinates.")
    parser.add_argument("--priority", type=int, default=0, help="Runtime scheduling priority.")
    parser.add_argument("--bpu-cores", nargs="+", type=int, default=None,
                        help="S-series BPU core indexes; X5 rejects this option.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--list-models", action="store_true", help="List manifest encoder/decoder assets without loading SDK.")
    mode.add_argument("--dry-run", action="store_true", help="Resolve paths and tensor contract without loading SDK or files.")
    return parser


def asset_ref(asset) -> str:
    """Render one asset reference for listing and reports."""
    return str(getattr(asset, "reference", getattr(asset, "asset_id", asset)))


def selection_report(selection) -> dict:
    """Serializable description of one resolved encoder/decoder selection."""
    return {
        "target": selection.target,
        "encoder_asset_id": asset_ref(selection.encoder_asset),
        "decoder_asset_id": asset_ref(selection.decoder_asset),
        "encoder_model_path": str(selection.encoder_model_path),
        "decoder_model_path": str(selection.decoder_model_path),
    }


def dry_run_report(selection, box) -> dict:
    """Static tensor-contract report; no files, SDK or image are touched."""
    report = selection_report(selection)
    report["input"] = {"shape": [1, 3, 512, 512], "dtype": "float32", "layout": "RGB NCHW"}
    report["decoder"] = {
        "embedding": [1, 256, 32, 32],
        "boxes_shapes": [[1, 4], [1, 4, 1, 1]] if selection.target == "x5" else [[1, 4]],
        "box_shape_status": "requires runtime metadata",
        "box": box, "mask_candidates": 3, "mask_semantics": "raw logits",
    }
    return report


def run_list_models(target: str) -> int:
    """Print the manifest encoder/decoder references for ``target`` (model-free)."""
    assets = list_available_assets(target)
    for asset in assets:
        print(asset_ref(asset))
    print(f"{len(assets)} manifest assets; no model loaded.")
    return 0


def read_bgr_image(path: "str | Path"):
    """Read one BGR image; failures name the exact path."""
    import cv2

    resolved = Path(path).expanduser()
    image = cv2.imread(str(resolved), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(str(resolved))
    return image


def save_outputs(result: dict, image, result_path: "str | Path", mask_path: "str | Path") -> None:
    """Write the overlay and binary-mask images (presentation only)."""
    import cv2
    import numpy as np


    overlay = draw_mask_result(image, result["mask"], result["iou"], result["mask_index"])
    result_file = Path(result_path).expanduser()
    mask_file = Path(mask_path).expanduser()
    result_file.parent.mkdir(parents=True, exist_ok=True)
    mask_file.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(result_file), overlay) or not cv2.imwrite(
        str(mask_file), result["mask"].astype(np.uint8) * 255
    ):
        raise OSError("failed to write MobileSAM output image")


def print_report(payload: dict) -> None:
    """Print the JSON result line for one finished prediction."""
    print(json.dumps(payload, indent=2))


__all__ = [
    "build_parser",
    "asset_ref",
    "dry_run_report",
    "parse_box",
    "print_report",
    "read_bgr_image",
    "run_list_models",
    "save_outputs",
    "selection_report",
]


def draw_mask_result(image: np.ndarray, mask: np.ndarray, iou: float, mask_index: int) -> np.ndarray:
    """Return a 512x512 BGR overlay; this function does not write files."""

    import cv2
    import numpy as np

    canvas = cv2.resize(image, (512, 512), interpolation=cv2.INTER_LINEAR).copy()
    mask_bool = np.asarray(mask, dtype=bool)
    if mask_bool.shape != (512, 512):
        raise ValueError(f"Expected a 512x512 mask, got {mask_bool.shape}.")
    color = np.zeros_like(canvas)
    color[:] = (0, 90, 220)
    blended = cv2.addWeighted(canvas, 0.45, color, 0.55, 0)
    canvas[mask_bool] = blended[mask_bool]
    contours, _ = cv2.findContours(mask_bool.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(canvas, contours, -1, (0, 255, 255), 2)
    for row, line in enumerate(("MobileSAM full mask: encoder + decoder", f"mask={mask_index}, IoU={float(iou):.4f}")):
        y = 28 + row * 28
        cv2.putText(canvas, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(canvas, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)
    return canvas
