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
"""RDK S Ultralytics YOLO inference entry point (compatibility wrapper).

The maintained implementation lives in the canonical sample at
`samples/vision/ultralytics_yolo/runtime/python/main.py`. This module keeps the
documented command line working from the RDK S tree, without a second copy of
the pipeline.

# The platform is resolved from the board, so S100, S100P and S600 select
# their own artifacts.
#
Usage:
    python3 main.py --task detect
    python3 main.py --task cls --model-path ../../model/nash-e/yolo11n_detect_nashe_640x640_nv12.hbm
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


_SAMPLE = _find_sample_root()
_MAIN = os.path.join(_SAMPLE, "runtime", "python", "main.py")


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
    """Forward to the canonical sample entry point on the RDK S series.

    Args:
        argv: Argument list, defaulting to `sys.argv[1:]`.

    Returns:
        None
    """
    # S100, S100P and S600 publish different artifacts, so the platform is
    # resolved from the board instead of being pinned here.
    _run(_MAIN, list(sys.argv[1:] if argv is None else argv))

if __name__ == "__main__":
    main()
