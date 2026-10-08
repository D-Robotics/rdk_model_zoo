"""FCOS command surface: options, listing, dry-run, labels, and rendering.

The entry point (``main.py``) parses arguments, constructs the runner and
task, and calls ``predict``; everything presentational lives here — the
parser with the published defaults, the model-free ``--list-models`` and
``--dry-run`` modes, label reading, and the annotated-image drawing.
Nothing in this module loads the board SDK, NumPy, or OpenCV at import time.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from samples.vision.fcos.runtime.python.model_binding import (
    SUPPORTED_TARGETS,
    list_available_assets,
)

ROOT = Path(__file__).resolve().parents[5]
SAMPLE_DIR = ROOT / "samples" / "vision" / "fcos"
DEFAULT_IMAGE = SAMPLE_DIR / "test_data" / "bus.jpg"
DEFAULT_LABELS = ROOT / "datasets" / "coco" / "coco_classes.names"
DEFAULT_RESULT = SAMPLE_DIR / "test_data" / "result.jpg"


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
