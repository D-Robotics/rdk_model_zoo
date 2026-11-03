# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Explicit manifest-backed Depth Anything V2 download; inference never downloads."""

import argparse
from pathlib import Path
from utils.py_utils.assets import download_asset
from samples.vision.depth_anything_v2.runtime.python.cli import (
    VARIANT_ASSETS,
    resolve_selection,
)


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Download the published S100 Depth Anything V2 HBM"
    )
    p.add_argument("--target", choices=("s100","s100p","s600"), default="s100")
    p.add_argument("--asset-id")
    p.add_argument("--variant", choices=tuple(VARIANT_ASSETS))
    p.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    args = p.parse_args(argv)
    s = resolve_selection(args.target, asset_id=args.asset_id, variant=args.variant)
    destination = args.output_dir / s.asset.filename
    digest = download_asset(s.asset, destination)
    print(f"Prepared {s.asset.reference} at {destination}; observed_sha256={digest}")
    if s.asset.sha256 is None:
        print(
            "Publisher SHA-256 is unknown; the observed digest is not independent provenance verification."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
