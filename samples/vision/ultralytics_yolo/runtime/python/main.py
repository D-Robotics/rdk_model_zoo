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

This file stays deliberately small: parse the arguments, resolve the plan,
construct the selected task model, call ``predict``, present the result —
the construct/predict/presentation sequence runs inline in ``main()``.
Option declarations, the model-free listing/dry-run/download-preparation
modes and result presentation live in ``yolo_cli.py``; each task's readable
flow lives in its task module (``detect.py`` for DFL detection).

One script drives every supported Ultralytics YOLO task on every supported
platform. The platform decides the artifact format, the download location,
the NV12 input tensor protocol and the documented default thresholds; the
task decides which wrapper runs.

The board runtime `hbm_runtime` is imported only when a model is actually
loaded. `--help`, `--list-models` and `--dry-run` therefore work on a
development host that has no board runtime installed.

Examples:
    python main.py --task detect --platform x5
    python main.py --task seg --platform s100 --family yolov8
    python main.py --list-models --platform s600
    python main.py --task detect --dry-run --platform x5 --family yolov8
    python main.py --task detect --model-path /path/to/custom.bin --platform x5
"""

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

from yolo_assets import UnsupportedAssetError  # noqa: E402
from yolo_platform import resolve_platform  # noqa: E402
from yolo_runtime import BoardRuntimeUnavailableError  # noqa: E402
import yolo_cli  # noqa: E402
from yolo_cli import (  # noqa: E402 - import paths kept for existing callers
    build_parser,  # the contract checker imports it from main
    describe_plan,
    ensure_model,
    load_labels,
    present_result,
    print_dry_run,
    print_model_listing,
)


def main() -> int:
    """Run the sample: resolve the plan, construct the model, predict once.

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

        # The readable flow itself: construct the dispatched task model,
        # run one prediction, present the result.
        from rdk_yolo_utils import file_io, inspect as inspect_utils
        from yolo_dispatch import create_runtime_model

        model = create_runtime_model(profile, args)
        model.set_scheduling_params(priority=args.priority, bpu_cores=args.bpu_cores)
        inspect_utils.print_model_info(model.model)
        image = file_io.load_image(args.test_img)
        result = model.predict(image)
        present_result(args, image, result, labels)
        return 0
    except (BoardRuntimeUnavailableError, UnsupportedAssetError, ValueError) as exc:
        print(f"[Error] {exc}", file=sys.stderr)
        return 2


def run_inference(profile, args, labels) -> None:
    """Compatibility entry for callers that resolved the plan themselves.

    Mirrors the construct → predict → present sequence ``main()`` performs
    inline (update the two together); the task-renderer tests patch the
    same helper seams both paths use.
    """
    from rdk_yolo_utils import file_io, inspect as inspect_utils
    from yolo_dispatch import create_runtime_model

    model = create_runtime_model(profile, args)
    model.set_scheduling_params(priority=args.priority, bpu_cores=args.bpu_cores)
    inspect_utils.print_model_info(model.model)
    image = file_io.load_image(args.test_img)
    result = model.predict(image)
    present_result(args, image, result, labels)


if __name__ == "__main__":
    sys.exit(main())
