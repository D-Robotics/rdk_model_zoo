#!/usr/bin/env python3
"""RDK X5 mapper entry point.

The conversion workflow is shared with the YOLO26 X5 mapper in
``workflow.py``.  This module only supplies the historical generic YOLO
defaults and keeps ``python mapper_x5.py`` working for existing users.
"""

from __future__ import annotations

from typing import Optional, Sequence

try:  # Imported as ``conversion.mapper_x5``.
    from .workflow import main_for_profile, x5_toolchain
except ImportError:  # Executed directly by the compatibility dispatcher.
    from workflow import main_for_profile, x5_toolchain


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Convert a generic YOLO ONNX model for RDK X5."""

    return main_for_profile(
        argv,
        x5_toolchain(),
        description="Convert an Ultralytics YOLO ONNX model for RDK X5",
    )


if __name__ == "__main__":
    raise SystemExit(main())
