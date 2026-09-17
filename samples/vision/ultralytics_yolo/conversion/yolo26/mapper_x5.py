#!/usr/bin/env python3
"""YOLO26 X5 mapper compatibility entry point.

YOLO26 changes the exported graph and runtime output protocol, but its X5
calibration and ``hb_mapper`` compiler protocol is the same as generic YOLO.
The shared workflow therefore owns both families; this file retains the old
path and the old calibration-directory default.
"""

from __future__ import annotations

from typing import Optional, Sequence

try:
    from ..workflow import main_for_profile, x5_toolchain
except ImportError:
    # The canonical dispatcher loads this file by path and puts conversion/
    # on sys.path before doing so.
    from workflow import main_for_profile, x5_toolchain


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Convert a YOLO26 ONNX model for RDK X5."""

    return main_for_profile(
        argv,
        x5_toolchain(),
        description="Convert a YOLO26 ONNX model for RDK X5",
        onnx_required=True,
        default_calibration_dir="calibration_data",
    )


if __name__ == "__main__":
    raise SystemExit(main())
