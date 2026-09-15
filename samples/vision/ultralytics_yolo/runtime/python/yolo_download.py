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

"""Download the Ultralytics YOLO model assets a platform publishes.

Every filename and URL comes from `yolo_assets`, which derives them from the
platform profile, so the downloader cannot drift from what the runtime
resolves. The script covers the whole published inventory of a platform, a
single family/task/size selection, the platform's documented default set, and a
dry run that only prints the plan.

`--dry-run` and `--help` never import the board runtime and never touch the
network.

Typical Usage:
    python yolo_download.py --platform x5
    python yolo_download.py --platform s600 --family yolov8 --task seg
    python yolo_download.py x5 yolo11 detect n        # legacy positional form
    python yolo_download.py --platform x5 --all --dry-run
"""

import argparse
import os
import sys

# Make the sample-local helper modules importable regardless of the working
# directory the sample is started from.
_PYTHON_DIR = os.path.dirname(os.path.abspath(__file__))
if _PYTHON_DIR not in sys.path:
    sys.path.insert(0, _PYTHON_DIR)

_SAMPLE_DIR = os.path.dirname(os.path.dirname(_PYTHON_DIR))
_DEFAULT_MODEL_DIR = os.path.join(_SAMPLE_DIR, "model")

from yolo_assets import (  # noqa: E402
    DEFAULT_FAMILY,
    DEFAULT_TASK,
    SUPPORTED_TASKS,
    UnsupportedAssetError,
    family_registry,
    model_filename,
    model_url,
)
from yolo_platform import (  # noqa: E402
    PlatformProfile,
    UnsupportedPlatformError,
    available_platforms,
    model_directory,
    resolve_platform,
)

#: Tasks the X5 tree downloaded when its download script ran with no argument.
X5_DEFAULT_TASKS = ('detect', 'seg', 'pose', 'cls')

#: Task the S tree downloaded when its download script ran with no argument.
S_DEFAULT_TASKS = ('detect',)


def default_tasks(profile: PlatformProfile) -> tuple:
    """Return the task set a platform's download script fetched by default.

    Args:
        profile: Platform whose documented default set is requested.

    Returns:
        A tuple of task names.
    """
    return X5_DEFAULT_TASKS if profile.family == "x5" else S_DEFAULT_TASKS


def iter_assets(profile: PlatformProfile):
    """Yield every asset a platform publishes.

    Args:
        profile: Platform whose inventory is enumerated.

    Yields:
        `(family, task, size)` triples.
    """
    for family, spec in family_registry(profile).items():
        for task in spec.tasks:
            for size in spec.task_sizes.get(task, spec.sizes):
                restricted = spec.size_platforms.get(size)
                if restricted is not None and profile.key not in restricted:
                    continue
                try:
                    model_filename(profile, family, task, size)
                except UnsupportedAssetError:
                    continue
                yield family, task, size


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser.

    Returns:
        The configured `argparse.ArgumentParser`.
    """
    parser = argparse.ArgumentParser(
        description="Download Ultralytics YOLO model assets for a platform.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="A legacy positional form is also accepted: "
               "download_model.sh [soc] [family] [task] [model_size].")
    parser.add_argument('--platform', type=str, default=None,
                        choices=list(available_platforms()),
                        help='Target platform. Defaults to the detected board.')
    parser.add_argument('--family', type=str, default=None,
                        help='Model family. Defaults to the platform default.')
    parser.add_argument('--task', type=str, default=None,
                        choices=list(SUPPORTED_TASKS),
                        help='Task. Downloads the platform default set when '
                             'omitted.')
    parser.add_argument('--model-size', type=str, default=None,
                        help='Model scale. Defaults to the family default.')
    parser.add_argument('--model-dir', type=str, default=_DEFAULT_MODEL_DIR,
                        help='Directory the sample stores models in.')
    parser.add_argument('--all', action='store_true',
                        help='Download every asset the platform publishes.')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print what would be downloaded, without '
                             'downloading anything.')
    parser.add_argument('positional', nargs='*',
                        help='Legacy positional form: soc family task size.')
    return parser


def apply_legacy_positionals(args, parser: argparse.ArgumentParser) -> None:
    """Map the legacy positional form onto the named options.

    The S download script took `[soc] [family] [task] [model_size]`
    positionally. That form keeps working, and an option given by name always
    wins over the positional equivalent.

    Args:
        args: Parsed command-line arguments, updated in place.
        parser: Parser used to report an invalid combination.

    Returns:
        None
    """
    if not args.positional:
        return
    if len(args.positional) > 4:
        parser.error("the legacy positional form takes at most "
                     "soc, family, task and model_size")
    names = ('platform', 'family', 'task', 'model_size')
    for name, value in zip(names, args.positional):
        if getattr(args, name) is None:
            setattr(args, name, value)
    # `--family` and `--task` are validated against the asset registry rather
    # than by argparse, because the legacy form may omit trailing fields.
    if args.task is not None and args.task not in SUPPORTED_TASKS:
        parser.error(f"unsupported task {args.task!r}; "
                     f"choose from {', '.join(SUPPORTED_TASKS)}")


def select_assets(args, profile: PlatformProfile):
    """Resolve the asset selections a set of arguments describes.

    Args:
        args: Parsed command-line arguments.
        profile: Selected platform.

    Returns:
        A list of `(family, task, size)` triples.

    Raises:
        UnsupportedAssetError: If the platform publishes no such asset.
    """
    if args.all:
        return list(iter_assets(profile))
    family = args.family or DEFAULT_FAMILY
    if args.task is None:
        return [(family, task, args.model_size)
                for task in default_tasks(profile)]
    return [(family, args.task, args.model_size)]


def main() -> int:
    """Run the downloader.

    Returns:
        A process exit status.
    """
    parser = build_parser()
    args = parser.parse_args()
    apply_legacy_positionals(args, parser)

    try:
        profile = resolve_platform(args.platform)
    except UnsupportedPlatformError as exc:
        print(f"[Error] {exc}", file=sys.stderr)
        return 2

    try:
        selections = select_assets(args, profile)
        targets = []
        for family, task, size in selections:
            filename = model_filename(profile, family, task, size)
            targets.append((
                filename,
                os.path.join(
                    model_directory(args.model_dir, profile), filename),
                model_url(profile, family, task, size),
            ))
    except UnsupportedAssetError as exc:
        print(f"[Error] {exc}", file=sys.stderr)
        return 2

    if not targets:
        print("[Error] Nothing to download.", file=sys.stderr)
        return 2

    print(f"Platform: {profile.key} ({profile.march}), "
          f"{len(targets)} asset(s) selected.")
    missing = [(name, path, url) for name, path, url in targets
               if not os.path.exists(path)]
    if args.dry_run:
        for name, path, url in targets:
            state = "present" if os.path.exists(path) else "missing"
            print(f"  [{state}] {path}\n            {url}")
        print(f"[dry-run] {len(missing)} of {len(targets)} asset(s) would be "
              f"downloaded. Nothing was downloaded.")
        return 0

    if not missing:
        print("All selected models are already present.")
        return 0

    from rdk_yolo_utils import file_io  # noqa: PLC0415 - keeps imports lazy

    for name, path, url in missing:
        print(f"[Download] {name}")
        file_io.download_model_if_needed(path, url)
    print(f"[Done] {len(missing)} model(s) downloaded.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
