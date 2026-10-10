# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Explicitly download original published YOLOE artifacts; never during inference."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from utils.py_utils.assets import download_asset
from samples.vision.yoloe.runtime.python.cli import resolve_selection


def build_parser():
    p = argparse.ArgumentParser(
        description="Download an exact published YOLOE model; S publications remain quantized."
    )
    p.add_argument("--target", required=True, choices=("x5", "s100", "s100p"))
    p.add_argument("--variant", default=None)
    p.add_argument("--asset-id", default=None)
    p.add_argument("--output-dir", type=Path, default=None)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        s = resolve_selection(args.target, variant=args.variant, asset_id=args.asset_id)
        path = (
            args.output_dir.expanduser() / s.asset.filename
            if args.output_dir is not None
            else s.model_path
        )
        digest = download_asset(s.asset, path)
        print(f"{s.asset.reference}\nPath: {path}\nObserved SHA-256: {digest}")
        if s.asset.sha256 is None:
            print("Publisher SHA-256 unknown; observed digest does not verify origin.")
        if not s.published_float:
            print(
                "Published S output is quantized; this download cannot run through the floating-only canonical entry."
            )
        return 0
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
