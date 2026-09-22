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

"""Ultralytics YOLO unified inference entry point.

One script drives every supported Ultralytics YOLO task on every supported
platform. The platform decides the artifact format, the download location, the
NV12 input tensor protocol and the documented default thresholds; the task
decides which wrapper runs.

The board runtime `hbm_runtime` is imported only when a model is actually
loaded. `--help`, `--list-models` and `--dry-run` therefore work on a
development host that has no board runtime installed.

Supported platforms:
    - x5:   RDK X5            (.bin, packed NV12,  bayes-e)
    - s100: RDK S100          (.hbm, split NV12,   nash-e)
    - s100p: RDK S100P        (.hbm, split NV12,   nash-m)
    - s600: RDK S600          (.hbm, split NV12,   nash-p)

Supported tasks:
    - detect: object detection (DFL head, or the NMS-free YOLOv10 head)
    - seg:    instance segmentation
    - pose:   pose estimation
    - cls:    image classification

Examples:
    python main.py --task detect --platform x5
    python main.py --task seg --platform s100 --family yolov8
    python main.py --list-models --platform s600
    python main.py --task detect --dry-run --platform x5 --family yolov8
    python main.py --task detect --model-path /path/to/custom.bin --platform x5
"""

import argparse
import os
import sys
from pathlib import Path

# Make the sample-local helper modules importable regardless of the working
# directory the sample is started from.
_PYTHON_DIR = os.path.dirname(os.path.abspath(__file__))
if _PYTHON_DIR not in sys.path:
    sys.path.insert(0, _PYTHON_DIR)

_REPOSITORY_ROOT = Path(__file__).resolve().parents[5]
if not (_REPOSITORY_ROOT / 'docs/release/platforms.json').is_file():
    raise RuntimeError('This entry requires a complete Model Zoo source checkout.')
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))
from samples._shared.platforms import resolve_target, require_execution_target

# Sample layout, resolved relative to this file rather than to the working
# directory, so the sample runs correctly from anywhere.
_SAMPLE_DIR = os.path.dirname(os.path.dirname(_PYTHON_DIR))
_MODEL_DIR = os.path.join(_SAMPLE_DIR, "model")
_TEST_DATA_DIR = os.path.join(_SAMPLE_DIR, "test_data")
_DEFAULT_IMAGE = os.path.join(_TEST_DATA_DIR, "bus.jpg")
_COCO_LABELS = os.path.join(_TEST_DATA_DIR, "coco_classes.names")
_IMAGENET_LABELS = os.path.join(_TEST_DATA_DIR, "imagenet_classes.names")
_OBB_LABELS = os.path.join(_TEST_DATA_DIR, "ultralytics_dota_classes.names")

import numpy as np  # noqa: E402 - imported after sys.path setup

from yolo_assets import (  # noqa: E402
    DEFAULT_FAMILY,
    DEFAULT_TASK,
    SUPPORTED_TASKS,
    UnsupportedAssetError,
    available_families,
    family_listing,
    manifest_asset,
    model_filename,
    model_path as resolve_model_path,
    model_url,
)
from yolo_platform import (  # noqa: E402
    PlatformProfile,
    UnsupportedPlatformError,
    available_platforms,
    resolve_platform,
)
from yolo_runtime import (  # noqa: E402
    BoardRuntimeUnavailableError,
    default_nms_thres,
    default_resize_type,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser.

    Returns:
        The configured `argparse.ArgumentParser`.
    """
    parser = argparse.ArgumentParser(
        description="Ultralytics YOLO unified inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--platform', '--target', type=str, default=None,
                        choices=['auto', *available_platforms()],
                        help='Target platform. When omitted, the board is '
                             'detected from /sys/class/boardinfo. Preparation '
                             'may select any target; inference requires matching '
                             'local board identity.')
    parser.add_argument('--task', type=str, default=DEFAULT_TASK,
                        choices=list(SUPPORTED_TASKS),
                        help='Task type.')
    parser.add_argument('--family', type=str, default=None,
                        help='Model family, for example yolo11, yolov8, '
                             'yolov10 or yolov5u.')
    parser.add_argument('--model-size', type=str, default=None,
                        help='Model scale, for example n, s, m, l, x. '
                             'Defaults to the family default of the platform.')
    parser.add_argument('--model-path', type=str, default=None,
                        help='Path to a compiled model. Overrides the resolved '
                             'default path and is never downloaded.')
    parser.add_argument('--asset-id', default=None,
                        help='Exact existing manifest reference group:sample:filename; '
                             'use --list-models to inspect references.')
    parser.add_argument('--input-shape', type=_input_shape, default=None,
                        help='Explicit model input geometry as HxW, used only '
                             'when the runtime does not report it.')
    parser.add_argument('--test-img', type=str, default=_DEFAULT_IMAGE,
                        help='Path to the test image.')
    parser.add_argument('--label-file', type=str, default=None,
                        help='Path to a label file. Defaults to the sample '
                             'COCO labels for detect/seg/pose and the sample '
                             'ImageNet labels for cls.')
    parser.add_argument('--img-save-path', type=str, default='result.jpg',
                        help='Path to save the rendered result image.')
    parser.add_argument('--score-thres', type=float, default=0.25,
                        help='Confidence score threshold.')
    parser.add_argument('--nms-thres', type=float, default=None,
                        help='IoU threshold for NMS. Defaults to the value the '
                             'selected platform documents.')
    parser.add_argument('--resize-type', type=int, default=None, choices=[0, 1],
                        help='Resize policy: 0 stretch, 1 letterbox. Defaults '
                             'to the value the selected platform documents.')
    parser.add_argument('--classes-num', type=int, default=None)
    parser.add_argument('--strides', type=lambda v: [int(x) for x in v.split(',')], default=[8,16,32])
    parser.add_argument('--mc', type=int, default=32)
    parser.add_argument('--angle-sign', type=float, default=1.0)
    parser.add_argument('--angle-offset', type=float, default=0.0, help='OBB offset in degrees')
    parser.add_argument('--regularize', type=int, choices=[0,1], default=1)
    parser.add_argument('--reg', type=int, default=16,
                        help='Number of DFL regression bins.')
    parser.add_argument('--nkpt', type=int, default=17,
                        help='[Pose] Number of keypoints.')
    parser.add_argument('--topk', type=int, default=5,
                        help='[Cls] Top-K classification results.')
    parser.add_argument('--kpt-conf-thres', type=float, default=0.50,
                        help='[Pose] Keypoint visibility threshold.')
    parser.add_argument('--priority', type=int, default=0,
                        help='Model priority (0~255). 0 is lowest, 255 highest.')
    parser.add_argument('--bpu-cores', nargs='+', type=int, default=[0],
                        help='BPU core indexes to run inference on.')
    parser.add_argument('--list-models', action='store_true',
                        help='List the model assets the selected platform '
                             'publishes, then exit. Needs no board runtime.')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print the resolved model path and download URL, '
                             'then exit without downloading or running. Needs '
                             'no board runtime.')
    parser.add_argument('--download', action='store_true',
                        help='Download the resolved model if it is missing, '
                             'then exit without running inference.')
    return parser


def _input_shape(value):
    """Parse an `HxW` input geometry override.

    Args:
        value: The raw `--input-shape` value, or `None`.

    Returns:
        A `(height, width)` tuple, or `None` when no override was given.

    Raises:
        argparse.ArgumentTypeError: If the value is not `HxW` with positive
            integers.
    """
    if value is None:
        return None
    parts = str(value).lower().replace('*', 'x').split('x')
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            f"--input-shape must be HxW, got {value!r}.")
    try:
        height, width = (int(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"--input-shape must be HxW with integers, got {value!r}.") from exc
    if height <= 0 or width <= 0:
        raise argparse.ArgumentTypeError(
            f"--input-shape must be positive, got {value!r}.")
    return height, width


def print_model_listing(profile: PlatformProfile) -> None:
    """Print the model assets a platform publishes.

    Args:
        profile: Platform to describe.

    Returns:
        None
    """
    print(f"Platform: {profile.key}")
    print(f"  artifact format : {profile.model_format}")
    print(f"  compiler march  : {profile.march}")
    print(f"  filename suffix : {profile.model_suffix}")
    print(f"  artifact dir    : "
          f"{os.path.join('model', profile.model_subdir) if profile.model_subdir else 'model'}")
    print(f"  NV12 input      : {profile.input_protocol}")
    print(f"  default NMS IoU : {profile.nms_thres}")
    print(f"  C++ runtime     : {'yes' if profile.supports_cpp else 'no'}")
    print("  published assets:")
    for entry in family_listing(profile):
        tasks = ", ".join(
            f"{task}({'/'.join(sizes)})"
            for task, sizes in entry["tasks"].items())
        note = f"  # {entry['notes']}" if entry["notes"] else ""
        print(f"    {entry['family']:<9} default size {entry['default_size']}"
              f"  ->  {tasks}{note}")
        for task, sizes in entry['tasks'].items():
            for size in sizes:
                record = manifest_asset(profile, entry['family'], task, size)
                print(f'      {task}: {record.reference} (asset available; validation is separate)')


def select_manifest_reference(profile: PlatformProfile, args) -> None:
    """Resolve an official reference using the sample's finite task/size policy."""
    from samples._shared.assets import resolve_asset
    try:
        record = resolve_asset(args.asset_id)
    except ValueError as exc:
        raise UnsupportedAssetError(str(exc)) from exc
    if record.group != profile.family or record.sample_id not in ('ultralytics_yolo', 'ultralytics_yolo26'):
        raise UnsupportedAssetError('Asset reference does not belong to this target and Sample.')
    candidates = []
    for entry in family_listing(profile):
        for size in entry['tasks'].get(args.task, ()):
            candidate = manifest_asset(profile, entry['family'], args.task, size)
            if candidate.reference == record.reference:
                candidates.append((entry['family'], size))
    if len(candidates) != 1:
        raise UnsupportedAssetError('Asset reference is not registered for the selected target/task.')
    family, size = candidates[0]
    if (args.family and args.family != family) or (args.model_size and args.model_size != size):
        raise UnsupportedAssetError('Asset reference conflicts with family or model size.')
    args.family, args.model_size = family, size


def describe_plan(profile: PlatformProfile, args) -> dict:
    """Resolve the model asset a set of arguments selects.

    Args:
        profile: Selected platform.
        args: Parsed command-line arguments.

    Returns:
        A dictionary with the resolved filename, local path, download URL and
        whether the artifact is already present. When `--model-path` was given
        the URL is `None`, because an explicit path is never downloaded.

    Raises:
        UnsupportedAssetError: If the platform publishes no such asset.
    """
    from yolo_assets import family_from_filename
    if args.asset_id:
        select_manifest_reference(profile, args)
    inferred = family_from_filename(profile, args.model_path) if args.model_path else None
    if args.family and inferred and args.family != inferred:
        raise UnsupportedAssetError("--family conflicts with --model-path filename.")
    args.family = args.family or inferred or DEFAULT_FAMILY
    from yolo_dispatch import get_task_types
    get_task_types(profile, args.family, args.task)
    if args.model_path:
        return {
            "asset_reference": args.asset_id,
            "filename": os.path.basename(args.model_path),
            "path": args.model_path,
            "url": None,
            "present": os.path.exists(args.model_path),
            "explicit": True,
        }
    filename = model_filename(
        profile, args.family, args.task, args.model_size)
    path = resolve_model_path(
        _MODEL_DIR, profile, args.family, args.task, args.model_size)
    return {
        "asset_reference": manifest_asset(profile, args.family, args.task, args.model_size).reference,
        "filename": filename,
        "path": path,
        "url": model_url(profile, args.family, args.task, args.model_size),
        "present": os.path.exists(path),
        "explicit": False,
    }


def print_dry_run(profile: PlatformProfile, args, plan: dict) -> None:
    """Print the resolved plan without touching the network or the board.

    Args:
        profile: Selected platform.
        args: Parsed command-line arguments.
        plan: The dictionary returned by `describe_plan`.

    Returns:
        None
    """
    from yolo_dispatch import runtime_resize

    print("[dry-run] No model is downloaded and no inference is run.")
    print(f"  platform        : {profile.key} ({profile.march})")
    print(f"  task            : {args.task}")
    print(f"  family          : {args.family}")
    print(f"  model file      : {plan['filename']}")
    if plan.get('asset_reference'):
        print(f"  asset reference : {plan['asset_reference']}")
    print(f"  resolved path   : {plan['path']}")
    print(f"  already present : {'yes' if plan['present'] else 'no'}")
    if plan['explicit']:
        print("  download url    : (not applicable, --model-path was given)")
    else:
        print(f"  download url    : {plan['url']}")
    print(f"  NV12 protocol   : {profile.input_protocol}")
    print(f"  resize policy   : "
          f"{args.resize_type if args.resize_type is not None else runtime_resize(profile, args.family, args.task)}")
    if args.task != 'cls':
        print(f"  NMS IoU         : "
              f"{args.nms_thres if args.nms_thres is not None else profile.nms_thres}")
    if profile.model_subdir:
        print(f"  artifact dir    : model/{profile.model_subdir}/")


def ensure_model(plan: dict) -> None:
    """Download the resolved model when it is missing.

    Args:
        plan: The dictionary returned by `describe_plan`.

    Returns:
        None

    Raises:
        FileNotFoundError: If an explicit model path does not exist, or if no
            download URL is available for a missing default asset.
    """
    from samples._shared.assets import resolve_asset, verify_asset_file, download_asset
    asset = resolve_asset(plan['asset_reference']) if plan.get('asset_reference') else None
    if plan['present']:
        if asset is not None:
            verify_asset_file(asset, Path(plan['path']))
        return
    if plan['explicit'] or not plan['url']:
        raise FileNotFoundError(f"Model file not found: {plan['path']}")
    print(f"[Download] {plan['url']}")
    if asset is None:
        raise ValueError('Default model download requires a manifest asset reference.')
    download_asset(asset, Path(plan['path']))


def load_labels(args, task: str) -> list:
    """Load the label names a task renders with.

    Args:
        args: Parsed command-line arguments.
        task: Task name.

    Returns:
        A list of label names, empty when no label file is available.

    Raises:
        FileNotFoundError: If an explicit `--label-file` does not exist.
    """
    from rdk_yolo_utils import file_io  # noqa: PLC0415 - keeps imports lazy

    if args.label_file:
        if not os.path.exists(args.label_file):
            raise FileNotFoundError(f"Label file not found: {args.label_file}")
        return file_io.load_class_names(args.label_file)
    if task == 'obb':
        return file_io.load_class_names(_OBB_LABELS)
    default = _IMAGENET_LABELS if task == 'cls' else _COCO_LABELS
    if os.path.exists(default):
        return file_io.load_class_names(default)
    return []


def run_inference(profile: PlatformProfile, args, labels: list) -> None:
    """One rendering pipeline for all supported families and their task protocols."""
    import cv2
    from rdk_yolo_utils import file_io, visualize, inspect as inspect_utils
    from yolo_dispatch import create_runtime_model
    model = create_runtime_model(profile,args)
    model.set_scheduling_params(priority=args.priority,bpu_cores=args.bpu_cores)
    inspect_utils.print_model_info(model.model)
    img=file_io.load_image(args.test_img)
    result=model.predict(img)
    result_img=None
    if args.task=='detect':
        boxes,scores,ids=result
        visualize.print_detections(boxes,scores,ids,labels)
        result_img=visualize.draw_boxes(img,boxes,ids,scores,labels,visualize.rdk_colors)
    elif args.task=='seg':
        boxes,scores,ids,masks=result
        visualize.draw_masks(img,boxes,masks,ids,visualize.rdk_colors)
        result_img=visualize.draw_boxes(img,boxes,ids,scores,labels,visualize.rdk_colors)
    elif args.task=='pose':
        boxes,scores,ids,xy,confidence=result
        kpts=np.concatenate([xy,confidence],axis=-1)
        result_img=visualize.draw_pose(img,boxes,kpts,kpt_conf_thres=args.kpt_conf_thres,scores=scores,class_ids=ids,colors=visualize.rdk_colors)
    elif args.task=='obb':
        result_img=img.copy()
        for item in result:
            cx,cy,w,h,angle=item['rrect']
            points=cv2.boxPoints(((float(cx),float(cy)),(float(w),float(h)),float(np.degrees(angle)))).astype(np.int32)
            color=tuple(int(v) for v in visualize.rdk_colors[int(item['id'])%len(visualize.rdk_colors)])
            cv2.polylines(result_img,[points],True,color,2)
            label=labels[item['id']] if 0<=item['id']<len(labels) else str(item['id'])
            cv2.putText(result_img,f"{label} {item['score']:.2f}",tuple(points[0]),cv2.FONT_HERSHEY_SIMPLEX,.5,color,1)
    else:
        visualize.print_classification_results(result,file_io.load_labels(args.label_file or _IMAGENET_LABELS))
    if result_img is not None:
        os.makedirs(os.path.dirname(os.path.abspath(args.img_save_path)),exist_ok=True)
        if not cv2.imwrite(args.img_save_path,result_img):raise OSError(f'Could not write image: {args.img_save_path}')
        print(f'[Saved] Result saved to: {args.img_save_path}')


def main() -> int:
    """Run the sample.

    Returns:
        A process exit status.
    """
    args = build_parser().parse_args()

    try:
        profile = resolve_platform(resolve_target(args.platform))
    except ValueError as exc:
        print(f"[Error] {exc}", file=sys.stderr)
        return 2

    if args.list_models:
        print_model_listing(profile)
        return 0

    try:
        plan = describe_plan(profile, args)
    except UnsupportedAssetError as exc:
        print(f"[Error] {exc}", file=sys.stderr)
        return 2

    if args.dry_run:
        print_dry_run(profile, args, plan)
        return 0

    if not args.download:
        try:
            require_execution_target(profile.key)
        except ValueError as exc:
            print(f'[Error] {exc}', file=sys.stderr)
            return 2

    try:
        ensure_model(plan)
    except (OSError, ValueError) as exc:
        print(f"[Error] {exc}", file=sys.stderr)
        return 2

    if args.download:
        print(f"[Ready] {plan['path']}")
        return 0

    args.model_path = plan['path']
    try:
        labels = load_labels(args, args.task)
        run_inference(profile, args, labels)
    except (BoardRuntimeUnavailableError, UnsupportedAssetError) as exc:
        print(f"[Error] {exc}", file=sys.stderr)
        return 2
    except ValueError as exc:
        print(f"[Error] {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
