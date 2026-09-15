#!/usr/bin/env python3

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

"""Run one evaluator over every compiled model in a directory.

This replaces the X5-only `eval_batch.py`, which walked `*.bin` files and
shelled out to a fixed evaluator script name. It now selects the evaluator
from the task embedded in each model file name and accepts both the X5 `.bin`
and the S `.hbm` suffixes.

The script prints the commands it is about to run and requires confirmation
unless `--yes` is given, because a full COCO or ImageNet pass takes hours.
"""

import argparse
import os
import subprocess
import sys

_EVALUATOR_DIR = os.path.dirname(os.path.abspath(__file__))
if _EVALUATOR_DIR not in sys.path:
    sys.path.insert(0, _EVALUATOR_DIR)

from eval_common import (  # noqa: E402  (path is set up above)
    add_platform_arguments,
    resolve_platform_argument,
)

#: Task token in a published file name to evaluator script.
TASK_EVALUATORS = {
    "detect": "eval_yolo_det.py",
    "seg": "eval_yolo_seg.py",
    "pose": "eval_yolo_pose.py",
    "cls": "eval_yolo_cls.py",
}

#: Model file suffixes the batch runner recognises.
MODEL_SUFFIXES = (".bin", ".hbm")


def evaluator_for(model_name: str):
    """Select the evaluator script for one model file name.

    Args:
        model_name: Model file name, for example
            `yolo11n_seg_bayese_640x640_nv12.bin`.

    Returns:
        The evaluator file name, or `None` when the task is not recognised.
    """
    stem = os.path.splitext(os.path.basename(model_name))[0].lower()
    for task, script in TASK_EVALUATORS.items():
        if f"_{task}_" in stem:
            return script
    return None


def build_parser() -> argparse.ArgumentParser:
    """Build the batch evaluator parser.

    Returns:
        The configured `argparse.ArgumentParser`.
    """
    parser = argparse.ArgumentParser(
        description="Run the matching evaluator over every compiled model in "
                    "a directory.")
    parser.add_argument("--model-dir", default=os.path.join(
        _EVALUATOR_DIR, os.pardir, "model"),
        help="Directory holding the compiled models.")
    parser.add_argument("--suffix", default="",
                        help="Extra suffix appended to each result JSON name.")
    parser.add_argument("--yes", action="store_true",
                        help="Skip the confirmation prompt.")
    add_platform_arguments(parser)
    return parser


def main(argv=None) -> int:
    """Dispatch one evaluator run per compiled model.

    Any option the batch parser does not own (for example `--image-dir` or
    `--annotation`) is forwarded verbatim to every evaluator run, so the same
    dataset arguments reach all of them.

    Args:
        argv: Argument list, defaulting to `sys.argv[1:]`.

    Returns:
        Process exit status; non-zero when a model could not be evaluated.
    """
    args, forwarded = build_parser().parse_known_args(
        sys.argv[1:] if argv is None else argv)
    platform = resolve_platform_argument(args)

    models = sorted(name for name in os.listdir(args.model_dir)
                    if name.lower().endswith(MODEL_SUFFIXES))
    if not models:
        print(f"[Error] no compiled model (*.bin or *.hbm) in "
              f"{args.model_dir}.", file=sys.stderr)
        return 1

    planned = []
    for name in models:
        script = evaluator_for(name)
        if script is None:
            print(f"[skip] {name}: no task token recognised.", file=sys.stderr)
            continue
        result = os.path.splitext(name)[0]
        if args.suffix:
            result = f"{result}_{args.suffix}"
        planned.append((name, script, os.path.join(args.model_dir,
                                                   f"{result}.json")))

    if not planned:
        print("[Error] no model file name carried a recognised task token.",
              file=sys.stderr)
        return 1

    for name, script, _json_path in planned:
        print(f"{name} -> {script}")

    if not args.yes:
        answer = input("[test] continue? (y/n) ")
        if answer.strip().lower() != "y":
            print("[stop] cancelled.")
            return 0

    status = 0
    for name, script, json_path in planned:
        command = [sys.executable, os.path.join(_EVALUATOR_DIR, script),
                   "--model-path", os.path.join(args.model_dir, name),
                   "--json-save-path", json_path,
                   "--platform", platform.key]
        command += forwarded
        print("[CMD] " + " ".join(command))
        status |= subprocess.call(command)
    return status


if __name__ == "__main__":
    sys.exit(main())
