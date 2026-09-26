# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Explicit manifest-backed download, separate from inference."""

import argparse
from pathlib import Path
import sys
from samples._shared.assets import download_asset
from samples.vision.yolo26_depth.runtime.python.model_binding import (
    TARGETS,
    VARIANTS,
    resolve_selection,
)


def main(argv=None):
    p = argparse.ArgumentParser(description="Download one published YOLO26 Depth model")
    p.add_argument("--target", choices=("auto",) + TARGETS, default="auto")
    p.add_argument("--variant", choices=VARIANTS)
    p.add_argument("--asset-id")
    p.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    args = p.parse_args(argv)
    try:
        s = resolve_selection(args.target, variant=args.variant, asset_id=args.asset_id)
        path = args.output_dir.expanduser() / s.asset.filename
        digest = download_asset(s.asset, path)
        print(f"Prepared {s.asset.reference} at {path}; observed_sha256={digest}")
        if s.asset.sha256 is None:
            print(
                "Publisher SHA-256 is unknown; observed digest is not independent provenance verification."
            )
        return 0
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
