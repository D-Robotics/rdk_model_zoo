#!/usr/bin/env python3
"""YOLO26 RDK S mapper compatibility entry point.

The YOLO26 ONNX graph is family-specific, while calibration preparation and
the Nash compiler protocol are shared with generic YOLO.  ``--march`` remains
the target-specific part of this adapter.
"""

from __future__ import annotations

from typing import Optional, Sequence

try:
    from ..workflow import main_for_profile, s_toolchain
except ImportError:
    from workflow import main_for_profile, s_toolchain


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Convert a YOLO26 ONNX model for an RDK S Nash target."""

    return main_for_profile(
        argv,
        s_toolchain(),
        description="Convert a YOLO26 ONNX model for RDK S",
        include_march=True,
    )


if __name__ == "__main__":
    raise SystemExit(main())
