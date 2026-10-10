"""FCOS command surface: options, listing, dry-run, labels, and rendering.

The entry point (``main.py``) parses arguments, constructs the runner and
task, and calls ``predict``; everything around selection and presentation
lives here — the published X5 asset identity and resolver, the parser with
the published defaults, the model-free ``--list-models`` and ``--dry-run``
modes, label reading, and the annotated-image drawing. Nothing in this
module loads the board SDK, NumPy, or OpenCV at import time. The tensor
contract and the readable task flow live in ``fcos.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from utils.py_utils.assets import Asset, list_assets

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
SAMPLE_DIR = ROOT / "samples" / "vision" / "fcos"
DEFAULT_IMAGE = SAMPLE_DIR / "test_data" / "bus.jpg"
DEFAULT_LABELS = ROOT / "datasets" / "coco" / "coco_classes.names"
DEFAULT_RESULT = SAMPLE_DIR / "test_data" / "result.jpg"


# ======================================================================
# Published FCOS asset identity and selection (docs/release/x5 rows via
# the shared asset reader; never inferred from an arbitrary filename).
# ======================================================================

SAMPLE = "fcos"


SUPPORTED_TARGETS = ("x5",)


STRIDES = (8, 16, 32, 64, 128)


CLASSES = 80


_VARIANT_FILES = {
    "fcos_efficientnetb0_detect_512x512_bayese_nv12.bin": "efficientnetb0",
    "fcos_efficientnetb2_detect_768x768_bayese_nv12.bin": "efficientnetb2",
    "fcos_efficientnetb3_detect_896x896_bayese_nv12.bin": "efficientnetb3",
}


_VARIANT_SIZES = {"efficientnetb0": 512, "efficientnetb2": 768, "efficientnetb3": 896}


class BindingError(ValueError):
    """A requested selection or runtime metadata violates the FCOS contract."""


@dataclass(frozen=True)
class AssetRecord:
    """One exact manifest-backed FCOS artifact."""

    target: str
    sample_id: str
    variant: str
    filename: str
    model_format: str
    url: str | None
    sha256: str | None

    @property
    def asset_id(self) -> str:
        """Return the exact qualified manifest identity."""
        return f"x5:{self.sample_id}:{self.filename}"

    @property
    def reference(self) -> str:
        """Alias for callers shared with other model samples."""
        return self.asset_id


@dataclass(frozen=True)
class FCOSContract:
    """Static geometry and output semantics for one published variant.

    FCOS follows the fixed source's ``dequantize_outputs`` behavior: a SCALE
    descriptor is applied to every observed dtype, including F32.  The runtime
    metadata must therefore carry a complete descriptor even for a float
    output; the binding never guesses a raw-float path from dtype alone.
    """

    variant: str
    input_height: int
    input_width: int
    classes_num: int = CLASSES
    strides: tuple[int, ...] = STRIDES
    resize_type: int = 0
    conf_thres: float = 0.5
    iou_thres: float = 0.6
    output_transform: str = "dequant"


@dataclass(frozen=True)
class ModelSelection:
    """Resolved artifact identity and its source-proven FCOS contract."""

    asset_id: str
    target: str
    variant: str
    model_path: Path
    contract: FCOSContract
    explicit_model_path: bool = False


def _records() -> tuple[AssetRecord, ...]:
    result = []
    for asset in list_assets("x5", SAMPLE):
        try:
            variant = _VARIANT_FILES[asset.filename]
        except KeyError as exc:
            raise BindingError(f"Unexpected FCOS manifest asset: {asset.filename!r}.") from exc
        result.append(AssetRecord("x5", SAMPLE, variant, asset.filename, asset.format, asset.url, asset.sha256))
    if {record.variant for record in result} != set(_VARIANT_SIZES):
        raise BindingError("FCOS manifest does not publish the required three variants.")
    return tuple(result)


def list_available_assets(target: str | None = None) -> tuple[AssetRecord, ...]:
    """List exact FCOS asset references for X5; ``auto`` is host-listing only."""
    if target not in (None, "auto", "x5"):
        raise BindingError(f"FCOS is published for x5 only, not {target!r}.")
    return _records()


def resolve_selection(
    target: str = "x5",
    *,
    asset_id: str | None = None,
    variant: str | None = None,
    model_path: str | Path | None = None,
) -> ModelSelection:
    """Resolve one exact manifest row and reject unqualified external paths."""
    if target == "auto":
        raise BindingError("FCOS execution requires explicit --target x5.")
    if target != "x5":
        raise BindingError(f"Unsupported FCOS target {target!r}; only x5 is published.")
    records = _records()
    if asset_id is None:
        requested_variant = "efficientnetb0" if variant is None else variant
        matches = [record for record in records if record.variant == requested_variant]
        if len(matches) != 1:
            raise BindingError("Select one FCOS variant with --variant or --asset-id.")
    else:
        matches = [record for record in records if record.asset_id == asset_id]
        if variant is not None:
            matches = [record for record in matches if record.variant == variant]
        if len(matches) != 1:
            raise BindingError(f"Unknown or mismatched FCOS asset_id {asset_id!r}.")
    record = matches[0]
    if model_path is not None and asset_id is None:
        raise BindingError("--model-path requires the exact --asset-id reference.")
    path = Path(model_path).expanduser() if model_path is not None else SAMPLE_DIR / "model" / record.filename
    size = _VARIANT_SIZES[record.variant]
    return ModelSelection(record.asset_id, target, record.variant, path, FCOSContract(record.variant, size, size), model_path is not None)


def build_parser() -> argparse.ArgumentParser:
    """Build the complete SDK-free FCOS CLI parser.

    Returns:
        The configured ``argparse.ArgumentParser``.
    """
    parser = argparse.ArgumentParser(description="FCOS object detection on RDK X5.")
    parser.add_argument("--target", choices=("auto",) + SUPPORTED_TARGETS, default="auto")
    parser.add_argument("--asset-id", default=None, help="Exact manifest asset reference.")
    parser.add_argument("--variant", choices=("efficientnetb0", "efficientnetb2", "efficientnetb3"), default=None, help="Variant; omitted selects B0 unless --asset-id selects another exact row.")
    parser.add_argument("--model-path", default=None, help="External model path; requires exact --asset-id.")
    parser.add_argument("--test-img", default=str(DEFAULT_IMAGE), help="BGR image path.")
    parser.add_argument("--label-file", default=str(DEFAULT_LABELS), help="COCO label file (optional for numeric output).")
    parser.add_argument("--img-save-path", default=str(DEFAULT_RESULT), help="Annotated image output path.")
    parser.add_argument("--resize-type", type=int, choices=(0, 1), default=None)
    parser.add_argument("--classes-num", type=int, default=80)
    parser.add_argument("--conf-thres", type=float, default=0.5)
    parser.add_argument("--iou-thres", type=float, default=0.6)
    parser.add_argument("--priority", type=int, default=0)
    parser.add_argument("--bpu-cores", nargs="+", type=int, default=[0])
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--list-models", action="store_true", help="List manifest assets without SDK access.")
    mode.add_argument("--dry-run", action="store_true", help="Resolve one exact selection without SDK or board access.")
    return parser


def list_models(target: str | None) -> int:
    """Print the manifest asset identities for ``target``.

    Args:
        target: Concrete target filter, ``auto``/None for all.

    Returns:
        int: 0 after printing the listing.
    """
    for record in list_available_assets(None if target == "auto" else target):
        print(record.asset_id)
    print("3 manifest assets; no model loaded.")
    return 0


def dry_run(selection, args) -> int:
    """Print the resolved selection and tensor contract without loading a model.

    Args:
        selection: Resolved ModelSelection from the binding module.
        args: Parsed namespace from :func:`build_parser`.

    Returns:
        int: 0 after printing the plan.
    """
    print(json.dumps({
        "target": selection.target,
        "asset_id": selection.asset_id,
        "variant": selection.variant,
        "model_path": str(selection.model_path),
        "input": {"shape": [1, 3, selection.contract.input_height, selection.contract.input_width], "dtype": "NV12", "layout": "packed"},
        "outputs": {"classification_heads": 5, "box_heads": 5, "center_heads": 5, "strides": list(selection.contract.strides)},
        "no_sdk_loaded": True,
    }, indent=2))
    return 0


def load_labels(path: Path) -> tuple[str, ...]:
    """Read one optional label file into ordered names.

    Args:
        path: Label file path; a missing file yields an empty tuple.

    Returns:
        The nonempty stripped label lines in file order.
    """
    if not path.is_file():
        return ()
    return tuple(line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def save_result(path: Path, image, result, labels: tuple[str, ...]) -> None:
    """Write the annotated detection image; the source image is not modified.

    Args:
        path: Destination image path; missing parent directories are created.
        image: uint8 BGR image shaped (H, W, 3).
        result: DetectionResult with boxes, scores, and class_ids.
        labels: Class names; out-of-range IDs render as numbers.

    Raises:
        OSError: When the image cannot be written.
    """
    import cv2

    canvas = image.copy()
    for box, score, class_id in zip(result.boxes, result.scores, result.class_ids):
        x1, y1, x2, y2 = [int(round(value)) for value in box]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 0, 255), 2)
        label = labels[int(class_id)] if int(class_id) < len(labels) else str(int(class_id))
        cv2.putText(canvas, f"{label} {float(score):.3f}", (x1, max(0, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), canvas):
        raise OSError(f"failed to write result image: {path}")
