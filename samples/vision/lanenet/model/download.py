# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Explicit manifest-backed LaneNet download; inference never downloads."""

import argparse
from pathlib import Path
from samples._shared.assets import download_asset
from samples.vision.lanenet.runtime.python.model_binding import (
    ASSET_ID,
    resolve_selection,
)


def main(argv=None):
    p = argparse.ArgumentParser(description="Download the published S100 LaneNet HBM")
    p.add_argument("--target", choices=("s100",), default="s100")
    p.add_argument("--asset-id", default=ASSET_ID)
    p.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    args = p.parse_args(argv)
    s = resolve_selection(args.target, asset_id=args.asset_id)
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
