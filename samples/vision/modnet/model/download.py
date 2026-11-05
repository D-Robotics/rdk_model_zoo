# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0

"""Prepare the legacy manual MODNet or an explicitly selected converted asset."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


ASSET_ID = "x5:modnet:modnet_512x512_rgb.bin"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Describe manual MODNet model preparation.")
    parser.add_argument("--target", choices=("x5","s100","s100p","s600"), default="x5")
    parser.add_argument("--asset-id")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    reference = args.asset_id or (ASSET_ID if args.target == 'x5' else None)
    if reference != ASSET_ID:
        from samples.vision.modnet.runtime.python.cli import resolve_selection
        from utils.py_utils.assets import download_asset
        try:
            selection = resolve_selection(args.target, asset_id=reference)
            digest = download_asset(selection.asset, args.output_dir / selection.asset.filename)
            print(f"Downloaded {selection.asset.reference}; SHA-256: {digest}")
            return 0
        except (OSError, ValueError) as exc:
            print(f"error: {exc}")
            return 2
    print(
        "MODNet is a manual asset: no public URL and sha256: null (unknown). "
        f"Obtain the external model and place it at {args.output_dir / 'modnet_512x512_rgb.bin'}; "
        f"runtime selection must use --asset-id {ASSET_ID}."
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
