# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0

"""Explicit LPRNet asset downloader; never invoked by runtime."""

from __future__ import annotations

import argparse
from pathlib import Path

from samples._shared.assets import download_asset, resolve_asset


ASSET_ID = "x5:lprnet:lpr.bin"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download the published LPRNet X5 model.")
    parser.add_argument("--target", choices=("x5",), default="x5")
    parser.add_argument("--asset-id", default=ASSET_ID)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    asset = resolve_asset(args.asset_id)
    destination = args.output_dir / asset.filename
    digest = download_asset(asset, destination)
    print(f"Prepared {asset.reference} at {destination} (observed_sha256={digest})")
    if asset.sha256 is None:
        print("Publisher sha256: null (unknown)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
