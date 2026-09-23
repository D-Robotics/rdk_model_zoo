# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Explicit manifest-backed FCOS artifact downloader."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from samples._shared.assets import download_asset, resolve_asset
from samples.vision.fcos.runtime.python.model_binding import list_available_assets

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent


def download_target(target: str, output_dir: str | Path | None = None, *, variant: str = "efficientnetb0", all_variants: bool = False) -> tuple[str, ...]:
    """Download one or all exact FCOS manifest rows; network is explicit here only."""
    records = list_available_assets(target)
    chosen = records if all_variants else tuple(record for record in records if record.variant == variant)
    if len(chosen) != (3 if all_variants else 1):
        raise ValueError(f"Unknown FCOS variant {variant!r} for target {target!r}.")
    root = Path(output_dir).expanduser() if output_dir is not None else DEFAULT_OUTPUT_DIR
    observed = []
    for record in chosen:
        asset = resolve_asset(record.asset_id)
        destination = root / record.filename
        digest = download_asset(asset, destination)
        observed.append(digest)
        print(f"Downloaded {record.asset_id} to {destination}; observed_sha256={digest}")
    return tuple(observed)


def build_parser() -> argparse.ArgumentParser:
    """Build the dependency-free downloader parser."""
    parser = argparse.ArgumentParser(description="Download manifest-backed FCOS X5 artifacts.")
    parser.add_argument("--target", choices=("x5",), default="x5")
    parser.add_argument("--variant", choices=("efficientnetb0", "efficientnetb2", "efficientnetb3"), default="efficientnetb0")
    parser.add_argument("--all", dest="all_variants", action="store_true", help="Download all three variants.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        download_target(args.target, args.output_dir, variant=args.variant, all_variants=args.all_variants)
        return 0
    except (OSError, ValueError) as exc:
        print(f"error: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
