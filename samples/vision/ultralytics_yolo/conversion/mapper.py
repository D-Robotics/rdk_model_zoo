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

"""Single entry point for the two Ultralytics YOLO conversion toolchains.

RDK X5 and the RDK S series do not share a compiler. X5 compiles with
`hb_mapper makertbin` and consumes raw float32 `.rgbchw` calibration data; the
S series compiles with `hb_compile` and consumes `/255`-normalized `.npy`
calibration data, with `quant_config` and `no_padding` extras. The two
implementations therefore stay separate:

    mapper_x5.py   X5 workflow, unchanged
    mapper_s.py    S workflow, unchanged

This module only selects between them from one `--platform` argument, and it
translates the platform into the matching `--march` for the S workflow. It
never merges the two calibration or quantization conventions.

The converter runs in the OpenExplore container, not on the board, and it never
installs packages: a missing dependency is reported instead.

Examples:
    python mapper.py --platform x5   --onnx ./yolo11n.onnx --cal-images ./cal_images
    python mapper.py --platform s100 --onnx ./yolo11n.onnx --cal-images ./cal_images
    python mapper.py --platform s600 --onnx ./yolo11n.onnx --cal-images ./cal_images

The exporter is separate and shared:
    python export_monkey_patch.py --pt ./yolo11n.pt --opset 11   # X5
    python export_monkey_patch.py --pt ./yolo11n.pt --opset 19   # S series
"""

import argparse
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

#: Platform name to the S workflow march string. The X5 workflow has a single
#: march, which it hard-codes.
S_MARCH = {
    "s100": "nash-e",
    "s100p": "nash-m",
    "s600": "nash-p",
}

#: Platform name to the opset the shared exporter should be run with.
RECOMMENDED_OPSET = {
    "x5": 11,
    "s100": 19,
    "s100p": 19,
    "s600": 19,
}

#: Arguments this dispatcher consumes itself and must not forward.
_DISPATCHER_OPTIONS = ("--platform", "--toolchain-help")


def build_parser() -> argparse.ArgumentParser:
    """Build the dispatcher parser.

    Returns:
        The configured `argparse.ArgumentParser`. Only the options this
        dispatcher owns are declared; everything else is forwarded verbatim to
        the selected toolchain, whose own `--help` documents it.
    """
    parser = argparse.ArgumentParser(
        description="Select the RDK X5 or RDK S Ultralytics YOLO conversion "
                    "toolchain. Unknown options are forwarded to it.",
        epilog="Recommended exporter opset: "
               + ", ".join(f"{key}={value}"
                           for key, value in RECOMMENDED_OPSET.items()))
    parser.add_argument('--platform', type=str, default=None,
                        choices=["x5"] + sorted(S_MARCH),
                        help='Target platform; selects the compiler toolchain. '
                             'Required unless a platform is selected by '
                             '--march when that is given.')
    parser.add_argument('--toolchain-help', action='store_true',
                        help='Show the selected toolchain own --help and exit.')
    parser.add_argument("--family", default="yolo11", help="Model family; yolo26 selects its conversion workflow.")
    return parser


def _option_value(argv, name: str):
    """Read the value of `name` from an argument list.

    Args:
        argv: Argument list that has not been fully parsed.
        name: Long option name, without the leading dashes.

    Returns:
        The option value, or `None` when the option is absent. A bare flag
        with no following value also yields `None`.
    """
    flag = f"--{name}"
    for index, arg in enumerate(argv):
        if arg == flag and index + 1 < len(argv):
            return argv[index + 1]
        if arg.startswith(f"{flag}="):
            return arg.split("=", 1)[1]
    return None


def resolve_platform(requested, forwarded) -> str:
    """Decide which toolchain a command line selects.

    An explicit `--platform` always wins. Otherwise a `--march` value selects
    the S toolchain for the matching architecture, and a command line with
    neither defaults to X5, which is what the single-toolchain X5 workflow
    used to be.

    Args:
        requested: Value of `--platform`, or `None`.
        forwarded: Remaining arguments, which may carry `--march`.

    Returns:
        One of the platform keys.

    Raises:
        ValueError: If `--march` names no supported architecture.
    """
    march = _option_value(forwarded, "march")
    if requested:
        if march is not None and march != S_MARCH.get(requested, "bayes-e"):
            raise ValueError("--platform and --march select different architectures.")
        return requested
    if march is None:
        raise ValueError("Pass --platform explicitly when converting on a host.")
    for platform, known_march in S_MARCH.items():
        if known_march == march:
            return platform
    raise ValueError(
        f"--march {march!r} matches no supported platform. Known "
        f"architectures: {', '.join(sorted(S_MARCH.values()))}.")


def main(argv=None) -> int:
    """Dispatch to the toolchain of the selected platform.

    Args:
        argv: Argument list, defaulting to `sys.argv[1:]`.

    Returns:
        The exit status of the selected toolchain.
    """
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    args, forwarded = parser.parse_known_args(argv)

    try:
        platform = resolve_platform(args.platform, forwarded)
    except ValueError as exc:
        print(f"[Error] {exc}", file=sys.stderr)
        return 2

    if platform == "x5":
        module_name = "mapper_x5"
    else:
        module_name = "mapper_s"
        # The S workflow takes the march as an argument; supply it from the
        # platform unless the caller already chose one.
        if _option_value(forwarded, "march") is None:
            forwarded += ['--march', S_MARCH[platform]]

    if args.family == 'yolo26':
        import importlib.util
        path=os.path.join(SCRIPT_DIR,'yolo26','mapper_x5.py' if platform=='x5' else 'mapper_s.py')
        spec=importlib.util.spec_from_file_location('_yolo26_mapper',path)
        module=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    else:
        module = __import__(module_name)
    if args.toolchain_help:
        saved = sys.argv
        sys.argv = [module_name] + forwarded + ['--help']
        try:
            module.main()
        except SystemExit:
            pass
        finally:
            sys.argv = saved
        return 0

    saved = sys.argv
    sys.argv = [module_name] + forwarded
    try:
        module.main()
    finally:
        sys.argv = saved
    return 0


if __name__ == "__main__":
    sys.exit(main())
