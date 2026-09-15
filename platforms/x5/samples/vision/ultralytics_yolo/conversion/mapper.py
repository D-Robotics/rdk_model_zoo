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
"""RDK X5 Ultralytics YOLO conversion entry point (compatibility wrapper).

The maintained implementation lives in the canonical sample at
`samples/vision/ultralytics_yolo/conversion/mapper.py`. This module forwards to
it from the RDK X5 tree, so the documented conversion command keeps working
without a second copy of the conversion workflow.

The two toolchains stay separate in the canonical sample: RDK X5 uses
`hb_mapper makertbin` with `.rgbchw` calibration data, the RDK S series uses
`hb_compile` with `/255`-normalised `.npy` calibration data. The toolchain the
arguments select is the one that runs.

# The X5 model is compiled with `hb_mapper makertbin` and `.rgbchw`
# calibration data; that toolchain is unchanged.
#
Usage:
    python3 mapper.py --help
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


_SCRIPT = os.path.join(_find_sample_root(), "conversion", "mapper.py")


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
    """Run the canonical conversion script.

    Args:
        argv: Argument list, defaulting to `sys.argv[1:]`.

    Returns:
        None
    """
    forwarded = list(sys.argv[1:] if argv is None else argv)
    # The X5 tree only ever produces `.bin` artifacts.
    if not _has_option(forwarded, "platform"):
        forwarded = ["--platform", "x5"] + forwarded
    _run(_SCRIPT, forwarded)


if __name__ == "__main__":
    main()
