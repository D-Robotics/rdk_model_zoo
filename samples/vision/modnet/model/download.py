# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0

"""Manual MODNet asset preparation; the manifest has no URL or publisher hash."""

from __future__ import annotations

import argparse
from pathlib import Path


ASSET_ID = "x5:modnet:modnet_512x512_rgb.bin"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Describe manual MODNet model preparation.")
    parser.add_argument("--target", choices=("x5",), default="x5")
    parser.add_argument("--asset-id", default=ASSET_ID)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.asset_id != ASSET_ID:
        print(f"error: expected exact asset-id {ASSET_ID}")
        return 2
    print(
        "MODNet is a manual asset: no public URL and sha256: null (unknown). "
        f"Obtain the external model and place it at {args.output_dir / 'modnet_512x512_rgb.bin'}; "
        f"runtime selection must use --asset-id {ASSET_ID}."
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
