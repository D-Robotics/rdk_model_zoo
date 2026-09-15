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
"""RDK X5 Ultralytics YOLO segmentation evaluation (compatibility wrapper).

The maintained implementation lives in the canonical sample at
`samples/vision/ultralytics_yolo/evaluator/eval_yolo_seg.py`. This module forwards to
it from the RDK X5 tree, so the documented evaluation command keeps working
without a second copy of the metric code.

# The pre-merge options `--image-path`, `--json-path`, `--max-num` and
# `--score-thres` are accepted and renamed onto the canonical options.
#
Usage:
    python3 eval_yolo_seg.py --help
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


_SCRIPT = os.path.join(_find_sample_root(), "evaluator", "eval_yolo_seg.py")


def _has_option(argv, name: str) -> bool:
    """Return True when `--name` appears as a flag or as `--name=value`.

    Args:
        argv: Argument list to scan.
        name: Option name without the leading dashes.

    Returns:
        True when the option is present.
    """
    flag = "--" + name
    return any(arg == flag or arg.startswith(flag + "=") for arg in argv)

#: Pre-merge X5 option names mapped onto the canonical evaluator options.
_RENAMED_OPTIONS = {
    "--image-path": "--image-dir",
    "--json-path": "--json-save-path",
    "--max-num": "--limit",
    "--score-thres": "--conf-thres",
}

#: Pre-merge options the merged evaluator derives instead of accepting.
_DERIVED_OPTIONS = {
    "--classes-num": "the class count is read from the model output",
    "--reg": "the DFL bin count is read from the model output",
    "--strides": "the feature strides are read from the model output",
    "--mc": "the mask coefficient count is read from the model output",
    "--nkpt": "the keypoint count is read from the model output",
    "--kpt-conf-thres": "keypoint confidence is reported per keypoint",
    "--result-image-dump": "the evaluator writes the result JSON only",
    "--result-image-path": "the evaluator writes the result JSON only",
    "--test-img": "the evaluator walks --image-dir instead",
    "--img-save-path": "the evaluator writes the result JSON only",
    "--is-open": "mask morphology follows the model wrapper default",
    "--is-point": "mask morphology follows the model wrapper default",
}


def _translate(argv: list) -> list:
    """Translate the pre-merge X5 evaluator options onto the canonical ones.

    Args:
        argv: Arguments as the caller passed them.

    Returns:
        Arguments the canonical evaluator accepts.
    """
    translated = []
    index = 0
    while index < len(argv):
        arg = argv[index]
        name = arg.split("=", 1)[0]
        if name in _DERIVED_OPTIONS:
            print(f"[notice] ignoring {name}: {_DERIVED_OPTIONS[name]}.",
                  file=sys.stderr)
            if "=" not in arg and index + 1 < len(argv):
                index += 1
            index += 1
            continue
        replacement = _RENAMED_OPTIONS.get(name)
        if replacement is None:
            translated.append(arg)
        elif "=" in arg:
            translated.append(replacement + arg[len(name):])
        else:
            translated.append(replacement)
        index += 1
    return translated



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

def main(argv=None) -> None:
    """Run the canonical evaluator.

    Args:
        argv: Argument list, defaulting to `sys.argv[1:]`.

    Returns:
        None
    """
    forwarded = list(sys.argv[1:] if argv is None else argv)
    forwarded = _translate(forwarded)
    _run(_SCRIPT, forwarded)


if __name__ == "__main__":
    main()
