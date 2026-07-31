#!/usr/bin/env python3
"""Reject synthetic calibration data and point users to the COCO workflow."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

DEFAULT_OUT = Path(__file__).resolve().parent.parent.parent / "calibration_data" / "images"


def main() -> int:
    """Exit without creating synthetic images."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    downloader = Path(__file__).with_name("download_coco_images.py")
    print(
        "Synthetic Vision calibration is disabled for this sample. "
        "Use the deterministic 50-image COCO val2017 set instead:\n"
        f"  python3 {downloader} --output-dir {args.output_dir}",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
