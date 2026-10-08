# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0

"""MODNet CLI options, published-model selection, listing, dry-run, and output.

``main.py`` uses these helpers to parse arguments, preview a selection, read
images, and save the matte/composite artifacts. The matting flow itself
lives in ``modnet.py``.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from utils.py_utils.assets import Asset, list_assets


class BindingError(ValueError):
    """A MODNet selection or metadata contract violation."""


SUPPORTED_TARGETS = ("x5", "s100", "s100p", "s600")
SAMPLE_DIR = Path(__file__).resolve().parents[2]
ASSET_ID = "x5:modnet:modnet_512x512_rgb.bin"

DEFAULT_TEST_IMAGE = SAMPLE_DIR / "test_data" / "person.jpg"
DEFAULT_BG_IMAGE = SAMPLE_DIR / "test_data" / "bg.jpg"
DEFAULT_MATTE_PATH = SAMPLE_DIR / "test_data" / "matte.png"
DEFAULT_RESULT_PATH = SAMPLE_DIR / "test_data" / "result.png"


@dataclass(frozen=True)
class ModelSelection:
    """One manual manifest asset and its selected local path.

    Attributes:
        target: Concrete execution target (``x5`` for MODNet).
        asset: Manifest asset record backing the selection.
        model_path: Local compiled model path.
        explicit_model_path: Whether the caller supplied the path explicitly.
    """

    target: str
    asset: Asset
    model_path: Path
    explicit_model_path: bool = False


def _asset() -> Asset:
    rows = list_assets("x5", "modnet")
    if len(rows) != 1 or rows[0].reference != ASSET_ID:
        raise BindingError(f"Expected one manifest asset {ASSET_ID!r}.")
    return rows[0]


def list_available_assets(target: Optional[str] = None) -> tuple[Asset, ...]:
    """List the manual X5 asset without board or SDK access.

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
    """Resolve MODNet's exact manual asset identity without loading it.

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
        raise BindingError("MODNet is published only for target x5.")
    asset = _asset()
    if asset_id is not None and asset_id != asset.reference:
        raise BindingError(f"Expected asset-id {asset.reference}, got {asset_id!r}.")
    if model_path is not None and asset_id is None:
        raise BindingError("An external model path requires --asset-id x5:modnet:modnet_512x512_rgb.bin.")
    return ModelSelection(
        target=key,
        asset=asset,
        model_path=Path(model_path).expanduser() if model_path else SAMPLE_DIR / "model" / asset.filename,
        explicit_model_path=model_path is not None,
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the host-safe MODNet CLI parser.

    Returns:
        argparse.ArgumentParser: Parser for selection, input, scheduling,
        output, and model-free listing/dry-run options.
    """
    parser = argparse.ArgumentParser(description="MODNet portrait matting")
    parser.add_argument("--target", choices=("auto",) + SUPPORTED_TARGETS, default="auto")
    parser.add_argument("--asset-id", help="Exact manifest reference, for example x5:modnet:modnet_512x512_rgb.bin")
    parser.add_argument("--model-path", help="External model path; requires the exact --asset-id")
    parser.add_argument("--test-img", default=str(DEFAULT_TEST_IMAGE), help="BGR input image")
    parser.add_argument("--bg-img", default=str(DEFAULT_BG_IMAGE), help="Optional background for composite output")
    parser.add_argument("--matte-save-path", default=str(DEFAULT_MATTE_PATH))
    parser.add_argument("--img-save-path", default=str(DEFAULT_RESULT_PATH))
    parser.add_argument("--priority", type=int, default=0)
    parser.add_argument("--bpu-cores", nargs="+", type=int, default=[0])
    parser.add_argument("--ref-size", type=int, default=512, help="Compiled model input size; must remain 512")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--list-models", action="store_true")
    modes.add_argument("--dry-run", action="store_true")
    return parser


def validate_scheduling(args: argparse.Namespace) -> None:
    """Reject scheduling values the run would refuse, including during dry-run.

    Args:
        args: Parsed namespace containing priority and bpu_cores.

    Returns:
        None.

    Raises:
        ValueError: Priority or a core index is out of range.
    """
    if type(args.priority) is not int or not 0 <= args.priority <= 255:
        raise ValueError("priority must be an integer between 0 and 255")
    if not args.bpu_cores or any(type(core) is not int or core < 0 for core in args.bpu_cores):
        raise ValueError("bpu-cores must be a non-empty list of non-negative integer indexes")


def run_list_models(target: str) -> int:
    """Print the manual MODNet asset as JSON without loading the SDK.

    Args:
        target: Concrete target or ``auto`` to list the publication.

    Returns:
        int: 0 after printing the list.
    """
    rows = list_available_assets(target)
    print(json.dumps([
        {"asset_id": row.reference, "filename": row.filename, "format": row.format,
         "url": row.url, "sha256": row.sha256, "target": "x5", "availability": "manual"}
        for row in rows
    ], ensure_ascii=False, indent=2))
    return 0


def run_dry_run(selection: ModelSelection) -> int:
    """Print a selection preview as JSON without loading the SDK.

    Args:
        selection: Resolved selection to preview.

    Returns:
        int: 0 after printing the preview.
    """
    print(json.dumps({
        "target": selection.target, "asset_id": selection.asset.reference,
        "model_path": str(selection.model_path), "model_format": selection.asset.format,
        "input_shape": [1, 3, 512, 512], "input_dtype": "float32",
        "output_shape": [1, 1, 512, 512], "output_dtype": "float32",
        "input_protocol": "BGR HWC -> RGB F32 NCHW, normalize [-1,1], long-side resize + zero padding",
        "model_path_exists": selection.model_path.is_file(),
        "sdk_loaded": False, "downloaded": False,
    }, ensure_ascii=False, indent=2))
    return 0


def read_image(path: "str | Path"):
    """Read one image as three-channel BGR pixels.

    Args:
        path: Image file path; leading ~ is expanded.

    Returns:
        np.ndarray: uint8 BGR array shaped (H, W, 3).

    Raises:
        FileNotFoundError: The path is missing or OpenCV cannot decode it.
    """
    import cv2

    image = cv2.imread(str(Path(path).expanduser()), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"input image not found or unreadable: {path}")
    return image


def save_matte(matte_save_path: "str | Path", matte) -> Path:
    """Write the predicted uint8 matte.

    Args:
        matte_save_path: Destination matte image path.
        matte: uint8 matte array to save.

    Returns:
        Path: Expanded destination path that was written.

    Raises:
        OSError: The destination directory or image cannot be written.
    """
    import cv2

    matte_path = Path(matte_save_path).expanduser()
    matte_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(matte_path), matte):
        raise OSError(f"failed to save matte: {matte_path}")
    return matte_path


def save_composite(img_save_path: "str | Path", composite) -> Path:
    """Write the optional composite result image.

    Args:
        img_save_path: Destination composite image path.
        composite: uint8 composite array to save.

    Returns:
        Path: Expanded destination path that was written.

    Raises:
        OSError: The destination directory or image cannot be written.
    """
    import cv2

    result_path = Path(img_save_path).expanduser()
    result_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(result_path), composite):
        raise OSError(f"failed to save composite: {result_path}")
    return result_path


def composite(image: np.ndarray, matte: np.ndarray, background: np.ndarray) -> np.ndarray:
    """Composite a uint8 matte over a resized BGR background."""

    import cv2
    import numpy as np

    if image.ndim != 3 or background.ndim != 3 or matte.shape != image.shape[:2]:
        raise ValueError("Image/background must be HWC and matte must match image geometry.")
    bg = cv2.resize(background, (image.shape[1], image.shape[0]))
    alpha = matte.astype(np.float32)[:, :, None] / 255.0
    return (image.astype(np.float32) * alpha + bg.astype(np.float32) * (1.0 - alpha)).astype(np.uint8)
