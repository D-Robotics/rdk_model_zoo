# Copyright (c) 2025 D-Robotics Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Shared command-line plumbing for the Ultralytics YOLO evaluators.

The four evaluators in this directory (`eval_yolo_det.py`, `eval_yolo_seg.py`,
`eval_yolo_pose.py`, `eval_yolo_cls.py`) share three decisions:

    - which platform profile the model was compiled for,
    - which score and NMS thresholds to use when the caller names none,
    - how a metric run with no predictions is reported.

Both original evaluator CLIs use NMS=0.70; runtime CLI defaults are separate.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional, Tuple

#: Image file suffixes the evaluators process.
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

#: Standard COCO model class-index to category-ID mapping.
COCO_CATEGORY_IDS: Tuple[int, ...] = (
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
    23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
    46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64,
    65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88,
    89, 90,
)

_EVALUATOR_DIR = os.path.dirname(os.path.abspath(__file__))
_RUNTIME_DIR = os.path.abspath(os.path.join(_EVALUATOR_DIR, os.pardir,
                                            "runtime", "python"))
if _RUNTIME_DIR not in sys.path:
    sys.path.insert(0, _RUNTIME_DIR)

_REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
if not (_REPOSITORY_ROOT / 'docs/release/platforms.json').is_file():
    raise RuntimeError('This entry requires a complete Model Zoo source checkout.')
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from yolo_platform import (  # noqa: E402  (path is set up above)
    PlatformProfile,
    UnsupportedPlatformError,
    resolve_platform,
)


def parse_input_shape(value: str) -> Tuple[int, int]:
    """Parse an `HxW` command-line value into a `(height, width)` tuple.

    Args:
        value: A string such as `"640x640"`.

    Returns:
        The parsed `(height, width)` tuple.

    Raises:
        argparse.ArgumentTypeError: If the value is not `HxW` with positive
            integers.
    """
    text = (value or "").strip().lower().replace("*", "x").replace(",", "x")
    parts = text.split("x")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            f"expected HxW such as 640x640, got {value!r}")
    try:
        height, width = (int(part) for part in parts)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"expected HxW such as 640x640, got {value!r}") from None
    if height <= 0 or width <= 0:
        raise argparse.ArgumentTypeError(
            f"expected positive HxW, got {value!r}")
    return height, width


def add_platform_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the platform and input-shape options to an evaluator parser.

    Args:
        parser: Parser to extend.

    Returns:
        None
    """
    parser.add_argument("--family", default=None, help="Infer from model filename when omitted.")
    parser.add_argument(
        "--platform", type=str, default=None,
        help="Target platform: x5, s100, s100p or s600. Defaults to the "
             "platform of the board the evaluator runs on. An explicit value "
             "always overrides board detection.")
    parser.add_argument(
        "--input-shape", type=parse_input_shape, default=None,
        help="Optional HxW override, for example 640x640. Only needed when "
             "the runtime does not report usable spatial input metadata.")


def add_threshold_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the score and NMS threshold options to an evaluator parser.

    Both default to `None`, meaning "use the value the platform profile
    documents". That keeps the X5 `0.70` and S `0.70` NMS defaults separate.

    Args:
        parser: Parser to extend.

    Returns:
        None
    """
    parser.add_argument("--classes-num", type=int, default=None)
    parser.add_argument("--strides", type=lambda s: [int(v) for v in s.split(",")], default=None)
    parser.add_argument("--kpt-conf-thres", type=float, default=0.5)
    parser.add_argument(
        "--conf-thres", type=float, default=None,
        help="Score threshold. Defaults to the model wrapper default (0.25).")
    parser.add_argument(
        "--nms-thres", type=float, default=0.70,
        help="IoU threshold for NMS. Defaults to the platform default: "
             "0.70 on RDK X5, 0.70 on the RDK S series.")


def resolve_platform_argument(args: argparse.Namespace) -> PlatformProfile:
    """Resolve the evaluator platform from the parsed arguments.

    Args:
        args: Parsed arguments carrying `platform`.

    Returns:
        The selected `PlatformProfile`.

    Raises:
        UnsupportedPlatformError: If no platform was given and the host board
            is not recognised, or if the requested platform is unknown.
    """
    return resolve_platform(getattr(args, "platform", None))


def list_images(image_dir: str,
                annotation: Optional[str] = None,
                limit: int = 0) -> list:
    """List the images an evaluation run should process.

    When an annotation file is given, its image list and image ids are
    authoritative, which is what COCO evaluation requires. Without one, the
    directory is walked in sorted order and each image is identified by its
    numeric filename, which is the convention used when they only
    dumped predictions.

    Args:
        image_dir: Directory holding the images.
        annotation: Optional COCO annotation JSON.
        limit: Maximum number of images; 0 means all.

    Returns:
        A list of `(image_id, file_name)` pairs.
    """
    if annotation:
        coco = _load_coco(annotation)
        image_ids = coco.getImgIds()
        if limit > 0:
            image_ids = image_ids[:limit]
        return [(int(image_id), coco.loadImgs([image_id])[0]["file_name"])
                for image_id in image_ids]
    names = sorted(name for name in os.listdir(image_dir)
                   if name.lower().endswith(IMAGE_SUFFIXES))
    if limit > 0:
        names = names[:limit]
    try:
        return [(int(os.path.splitext(name)[0]), name) for name in names]
    except ValueError as exc:
        raise ValueError("Non-numeric image names require --annotation for image IDs.") from exc


def category_ids(annotation: Optional[str] = None) -> list:
    """Map the standard COCO model indices, independent of annotation subsets.

    The annotation argument is retained for callers; it does not redefine the
    model's class order. Custom class-order models need a separate evaluator.
    """
    return list(COCO_CATEGORY_IDS)


def _load_coco(annotation: str):
    """Import and construct a `pycocotools` COCO object lazily.

    Args:
        annotation: Path of the COCO annotation JSON.

    Returns:
        The constructed `COCO` object.
    """
    from pycocotools.coco import COCO  # noqa: PLC0415 - heavy, optional import

    return COCO(annotation)


def report_empty_predictions(task: str, path: str) -> None:
    """Report a metric run that produced no predictions.

    An empty prediction file is a valid evaluation outcome, not a crash: it
    happens when every image scores below the threshold. The JSON file is
    still written so the run is auditable, and the caller skips the COCO or
    ImageNet metric computation, which this implementation skips for an empty result.

    Args:
        task: Task name, used in the message.
        path: Path of the written prediction file.

    Returns:
        None
    """
    print(f"[{task}] no predictions were produced; wrote an empty result "
          f"file to {path}. This evaluator skipped metric computation; "
          f"this does not mean AP is undefined or that the run passed.")


__all__ = [
    "COCO_CATEGORY_IDS",
    "IMAGE_SUFFIXES",
    "UnsupportedPlatformError",
    "add_platform_arguments",
    "add_threshold_arguments",
    "category_ids",
    "list_images",
    "parse_input_shape",
    "report_empty_predictions",
    "resolve_platform_argument",
]


def evaluation_types(args, platform, task):
    from yolo_assets import family_from_filename
    from yolo_dispatch import get_task_types
    inferred=family_from_filename(platform,args.model_path)
    if args.family and inferred and args.family!=inferred:
        raise ValueError('--family conflicts with model filename.')
    family=args.family or inferred or 'yolo11'
    if family=='yolo11' and not inferred and not args.family:
        print(f'[warn] cannot infer the model family from {args.model_path!r}; '
              'falling back to the yolo11 DFL decoder. Pass --family explicitly '
              'if the artifact belongs to another family.')
    return get_task_types(platform,family,task)


def evaluation_options(args,task):
    options={}
    if getattr(args,'strides',None) is not None:options['strides']=args.strides
    if task in ('detect','seg','obb') and getattr(args,'classes_num',None) is not None:options['classes_num']=args.classes_num
    return options
