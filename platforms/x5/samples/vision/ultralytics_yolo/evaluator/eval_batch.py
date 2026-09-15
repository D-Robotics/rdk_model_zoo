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
"""RDK X5 batch evaluation compatibility entry point.

The maintained implementation lives in the canonical sample at
`samples/vision/ultralytics_yolo/evaluator/eval_batch.py`. The pre-merge X5
script took `--eval-script`, `--bin-paths` and `--str`; the canonical script
selects the evaluator from each model file name, so only `--bin-paths` needs
translating, to `--model-dir`. The other two are accepted and ignored with a
notice, so existing scripted invocations keep working.

Usage:
    python3 eval_batch.py --bin-paths ../../model
"""

import os
import sys


def _find_sample_root() -> str:
    """Locate the canonical Ultralytics YOLO sample above this file.

    The platform trees are distributions of the merged sample, so the
    canonical copy always sits at `<repo>/samples/vision/ultralytics_yolo`.
    Walking up to it instead of counting `..` keeps this working when the
    sample is moved, and turns a missing canonical sample into an explicit
    error instead of a confusing import failure.

    A directory only counts as the canonical sample when it carries the shared
    downloader, because every platform tree ends in the same
    `samples/vision/ultralytics_yolo` tail and would otherwise match first.

    Returns:
        Absolute path of the canonical sample directory.

    Raises:
        ImportError: If no parent directory holds the canonical sample.
    """
    current = os.path.dirname(os.path.abspath(__file__))
    while True:
        candidate = os.path.join(current, "samples", "vision",
                                 "ultralytics_yolo")
        if os.path.isfile(os.path.join(candidate, "runtime", "python",
                                       "yolo_download.py")):
            return candidate
        parent = os.path.dirname(current)
        if parent == current:
            raise ImportError(
                "the canonical Ultralytics YOLO sample was not found above "
                f"{os.path.dirname(os.path.abspath(__file__))}; this "
                "compatibility entry point forwards to it and cannot run "
                "without it.")
        current = parent


_SCRIPT = os.path.join(_find_sample_root(), "evaluator", "eval_batch.py")

#: Pre-merge X5 options the canonical batch runner replaces by deriving the
#: evaluator and the result name from each model file name.
_DROPPED_OPTIONS = {
    "--eval-script": "the evaluator is selected from each model file name",
    "--str": "the result file name is derived from each model file name",
}


def main(argv=None) -> None:
    """Forward to the canonical batch evaluator.

    Args:
        argv: Argument list, defaulting to `sys.argv[1:]`.

    Returns:
        None
    """
    forwarded = []
    source = list(sys.argv[1:] if argv is None else argv)
    index = 0
    while index < len(source):
        arg = source[index]
        name = arg.split("=", 1)[0]
        if name == "--bin-paths":
            forwarded.append("--model-dir" + arg[len(name):])
        elif name in _DROPPED_OPTIONS:
            print(f"[notice] ignoring {name}: {_DROPPED_OPTIONS[name]}.",
                  file=sys.stderr)
            if "=" not in arg and index + 1 < len(source):
                index += 1
            index += 1
            continue
        else:
            forwarded.append(arg)
        index += 1

    _run(_SCRIPT, forwarded)



def _run(script: str, argv: list) -> None:
    """Run a canonical script as `__main__` with the given arguments.

    Args:
        script: Absolute path of the canonical script to run.
        argv: Arguments the script should see.

    Returns:
        None
    """
    import runpy  # noqa: PLC0415 - only needed when the wrapper actually runs

    sys.argv = [script] + list(argv)
    runpy.run_path(script, run_name="__main__")

if __name__ == "__main__":
    main()
