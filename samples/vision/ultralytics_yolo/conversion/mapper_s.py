#!/usr/bin/env python3
"""RDK S mapper entry point for S100, S100P, and S600.

Calibration preparation, configuration rendering, artifact movement, and
compiler execution are shared with the generic X5 and YOLO26 mappers.  The
selected Nash ``--march`` remains a target-specific compiler parameter.
"""

from __future__ import annotations

from typing import Optional, Sequence

try:  # Imported as ``conversion.mapper_s``.
    from .workflow import main_for_profile, s_toolchain
except ImportError:  # Executed directly by the compatibility dispatcher.
    from workflow import main_for_profile, s_toolchain


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Convert a generic YOLO ONNX model for an RDK S Nash target."""

    # The parser uses nash-e as the historical S100 default.  The dispatcher
    # supplies a platform-derived march when the caller selected S100P/S600.
    return main_for_profile(
        argv,
        s_toolchain(),
        description="Convert an Ultralytics YOLO ONNX model for RDK S",
        include_march=True,
    )


if __name__ == "__main__":
    raise SystemExit(main())
