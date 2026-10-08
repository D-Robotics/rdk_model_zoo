"""YOLOWorld command surface: options, listing, dry-run, and result rendering.

The entry point (``main.py``) parses arguments, constructs the runner and
task, and calls ``predict``; everything presentational lives here — the
parser with the published defaults, the model-free ``--list-models`` and
``--dry-run`` modes, prompt parsing, and the annotated-image drawing.
Nothing in this module loads the board SDK, NumPy, or OpenCV at import time.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from samples.vision.yoloworld.runtime.python.model_binding import (
    SAMPLE_DIR,
    list_available_assets,
    resolve_selection,
)


def parse_prompts(value: str) -> list[str]:
    """Split one comma-separated prompt list with the source validation rules.

    Args:
        value: Raw ``--prompts`` string.

    Returns:
        The stripped nonempty prompts.

    Raises:
        ValueError: On empty entries or more than 32 prompts.
    """
    pieces = [part.strip() for part in value.split(',')]
    if not pieces or any(not part for part in pieces):
        raise ValueError('Prompts must contain one or more nonempty comma-separated words; empty prompts are rejected.')
    if len(pieces) > 32:
        raise ValueError('At most 32 prompts are supported.')
    return pieces


def build_parser():
    """Build the YOLOWorld command-line parser with its published defaults.

    Returns:
        The configured ``argparse.ArgumentParser``.
    """
    parser = argparse.ArgumentParser(description='YOLOWorld X5 open-vocabulary RGB + offline text detection.')
    parser.add_argument('--target', choices=('auto', 'x5', 's100', 's100p', 's600'), default='auto')
    parser.add_argument('--model-path', default=None, help='Explicit yolo_world.bin path; requires exact --asset-id.')
    parser.add_argument('--asset-id', default=None, help='Exact manifest identity x5:yoloworld:yolo_world.bin.')
    parser.add_argument('--vocab-file', default=str(SAMPLE_DIR / 'test_data/offline_vocabulary_embeddings.json'))
    parser.add_argument('--test-img', default=str(SAMPLE_DIR / 'test_data/dog.jpeg'))
    parser.add_argument('--prompts', default='dog', help='Comma-separated vocabulary entries; empty entries are rejected.')
    parser.add_argument('--img-save-path', default=str(SAMPLE_DIR / 'test_data/inference.png'))
    parser.add_argument('--priority', type=int, default=0)
    parser.add_argument('--bpu-cores', type=int, nargs='+', default=[0])
    parser.add_argument('--score-thres', type=float, default=0.05)
    parser.add_argument('--nms-thres', type=float, default=0.45)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--list-models', action='store_true')
    modes.add_argument('--dry-run', action='store_true')
    return parser


def list_models(target: str) -> int:
    """Print the published YOLOWorld asset references for ``target``.

    Args:
        target: Concrete target or ``auto``.

    Returns:
        int: 0 after printing, including for an empty result.
    """
    for asset in list_available_assets(target):
        print(asset.reference)
    return 0


def dry_run(selection, prompts, args) -> int:
    """Print the resolved selection and tensor protocol without loading a model.

    Args:
        selection: Resolved ModelSelection.
        prompts: Parsed prompt list.
        args: Parsed namespace from :func:`build_parser`.

    Returns:
        int: 0 after printing the plan.
    """
    print(json.dumps({'target': selection.target, 'asset_id': selection.asset.reference,
                      'model_path': str(selection.model_path),
                      'image_input': 'float32[1,3,640,640] RGB, longest-side resize and top-left zero pad',
                      'text_input': 'float32[1,32,512,1] offline embeddings, last prompt fills slots',
                      'outputs': ['float32[1,8400,32] class scores', 'float32[1,8400,4] boxes'],
                      'prompts': prompts, 'score_thres': args.score_thres, 'nms_thres': args.nms_thres}, indent=2))
    return 0


def draw_results(image, result, class_names):
    """Draw one owned annotated BGR copy of ``image``.

    Args:
        image: uint8 BGR image shaped (H, W, 3); not modified.
        result: DetectionResult with boxes, scores, class_ids, and prompts.
        class_names: Offline vocabulary names indexed by class ID.

    Returns:
        The annotated copy of ``image``.
    """
    import cv2

    output = image.copy()
    for box, score, cid in zip(result.boxes, result.scores, result.class_ids):
        x1, y1, x2, y2 = map(int, box)
        color = (0, 180, 255)
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        name = class_names[int(cid)] if 0 <= int(cid) < len(class_names) else str(int(cid))
        cv2.putText(output, f"{name}: {float(score):.3f}", (x1, max(0, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, .6, color, 2)
    return output


def save_image(path, image) -> None:
    """Write ``image`` to ``path``, creating parent directories.

    Args:
        path: Destination image path.
        image: BGR image to encode.

    Raises:
        OSError: When the image cannot be written.
    """
    import cv2

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Could not save {path}")
