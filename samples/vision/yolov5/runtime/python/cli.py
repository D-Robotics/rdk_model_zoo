"""YOLOv5 command surface: options, listing, dry-run, and result rendering.

The entry point (``main.py``) parses arguments, constructs the runner and
task, and calls ``predict``; everything presentational lives here — the
parser with the published defaults, the model-free ``--list-models`` and
``--dry-run`` modes, option validation, and the annotated-image drawing.
Nothing in this module loads the board SDK, NumPy, or OpenCV at import time.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from samples.vision.yolov5.runtime.python.model_binding import (
    ANCHORS,
    SAMPLE_DIR,
    STRIDES,
    list_available_assets,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the YOLOv5 command-line parser with its published defaults.

    Returns:
        argparse.ArgumentParser: Parser for target/variant selection, input,
        thresholds, resize policy, scheduling, and the model-free
        listing/dry-run modes.
    """
    parser = argparse.ArgumentParser(description="YOLOv5 customer CLI: explicit model preparation, inference and rendering.")
    parser.add_argument('--target', choices=('auto', 'x5', 's100', 's100p', 's600'), default='auto')
    parser.add_argument('--variant', help='X5: n/s/m/l/x-v7.0 or s/m/l/x-v2.0; S: x-672. Omitted uses n-v7.0 / x-672.')
    parser.add_argument('--asset-id')
    parser.add_argument('--model-path')
    parser.add_argument('--test-img', help='Default X5 bus.jpg; S kite.jpg from sample test_data.')
    parser.add_argument('--label-file', default=str(SAMPLE_DIR / 'test_data/coco_classes.names'))
    parser.add_argument('--img-save-path', default=str(SAMPLE_DIR / 'test_data/result_unified.jpg'))
    parser.add_argument('--score-thres', type=float, default=.25)
    parser.add_argument('--nms-thres', type=float, default=.45)
    parser.add_argument('--resize-type', type=int, choices=(0, 1), default=None, help='Omitted: X5 stretch(0), S letterbox(1).')
    parser.add_argument('--classes-num', type=int, choices=(80,), default=80, help='Published assets bind exactly 80 classes.')
    parser.add_argument('--anchors', type=lambda s: tuple(float(v) for v in s.split(',')), default=ANCHORS)
    parser.add_argument('--strides', type=lambda s: tuple(int(v) for v in s.split(',')), default=STRIDES, help='Published heads require 8,16,32.')
    parser.add_argument('--priority', type=int, default=0)
    parser.add_argument('--bpu-cores', nargs='+', type=int, default=[0])
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--list-models', action='store_true')
    modes.add_argument('--dry-run', action='store_true')
    return parser


def list_models(target: str) -> int:
    """Print the published YOLOv5 asset references for ``target``.

    Args:
        target: Concrete target or ``auto`` to list every published target.

    Returns:
        int: 0 after printing, including for an empty result.
    """
    for asset in list_available_assets(target):
        print(asset.reference)
    return 0


def validate_options(args) -> None:
    """Reject option values the run would refuse, before any model work.

    Args:
        args: Parsed namespace from :func:`build_parser`.

    Raises:
        ValueError: A dry-run without an explicit target, thresholds outside
        [0,1], invalid scheduling values, or strides/anchors that do not
        match the published heads.
    """
    if args.dry_run and args.target == 'auto':
        raise ValueError('--dry-run requires an explicit --target.')
    if any(not math.isfinite(v) or not 0 <= v <= 1 for v in (args.score_thres, args.nms_thres)):
        raise ValueError('Thresholds must be finite in [0,1].')
    if not 0 <= args.priority <= 255 or any(v < 0 for v in args.bpu_cores):
        raise ValueError('Invalid scheduling priority/core index.')
    if tuple(args.strides) != STRIDES:
        raise ValueError('Published heads require strides 8,16,32.')
    if len(args.anchors) != 18 or any(not math.isfinite(v) or v <= 0 for v in args.anchors):
        raise ValueError('anchors require 18 finite positive numbers.')


def dry_run(selection, image_path: Path) -> int:
    """Print the resolved selection without downloading or loading the SDK.

    Args:
        selection: ModelSelection from the binding module.
        image_path: Image the run would use after default resolution.

    Returns:
        int: 0 after printing the plan.
    """
    print(json.dumps(dict(target=selection.target, variant=selection.variant,
                          asset_id=selection.asset.reference, model_path=str(selection.model_path),
                          test_img=str(image_path),
                          input_protocol='packed_nv12_640' if selection.target == 'x5' else 'split_nv12_672',
                          metadata_status='requires runtime metadata', board_status='not-run'), indent=2))
    return 0


def default_image_path(args, selection) -> Path:
    """Return the explicit or per-target default test image.

    Args:
        args: Parsed namespace from :func:`build_parser`.
        selection: Resolved model selection supplying the target.

    Returns:
        Path: Expanded explicit path, or bus.jpg (X5) / kite.jpg (S).
    """
    if args.test_img:
        return Path(args.test_img).expanduser()
    return SAMPLE_DIR / 'test_data' / ('bus.jpg' if selection.target == 'x5' else 'kite.jpg')


def draw_detections(image, result, labels):
    """Draw one owned annotated BGR copy; source image and result are unchanged.

    Args:
        image: uint8 BGR image shaped (H, W, 3); not modified.
        result: DetectionResult with boxes, scores, and class_ids.
        labels: Class-name sequence; out-of-range IDs render as numbers.

    Returns:
        The annotated copy of ``image``.
    """
    import cv2

    output = image.copy()
    for box, score, class_id in zip(result.boxes, result.scores, result.class_ids):
        x1, y1, x2, y2 = (int(v) for v in box)
        label = labels[int(class_id)] if 0 <= class_id < len(labels) else str(class_id)
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 200, 0), 2)
        cv2.putText(output, f'{label}: {score:.3f}', (x1, max(15, y1)), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 200, 0), 1)
    return output
