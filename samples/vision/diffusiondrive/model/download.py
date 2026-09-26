# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Explicit exact-target downloads with published SHA-256 verification."""

import argparse
from pathlib import Path
from samples._shared.assets import download_asset
from samples.vision.diffusiondrive.runtime.python.model_binding import (
    resolve_selection,
    SAMPLE_DIR,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=("s100p", "s600"), required=True)
    parser.add_argument("--asset-id")
    parser.add_argument("--output-dir", type=Path, default=SAMPLE_DIR / "model")
    args = parser.parse_args(argv)
    selection = resolve_selection(args.target, asset_id=args.asset_id)
    destination = args.output_dir.expanduser() / selection.asset.filename
    digest = download_asset(selection.asset, destination)
    print(f"Prepared {selection.asset.reference} at {destination}; sha256={digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
