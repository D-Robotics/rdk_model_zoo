#!/usr/bin/env python3
# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Download the published PartA/PartB pairs into the model directory.

Each target's pair is fetched from the release manifest
(``docs/release/{x5,s}/models.yaml``) with the shared asset downloader,
which verifies the publisher SHA-256 before installing the file and never
overwrites an existing artifact.  The manifest digests are additionally
cross-checked against ``model/frcnn_mbv3_320.json`` so the published
registry and the runtime descriptor can never silently drift apart.

The default destination is the runtime's own lookup path
(``model/<target>/``); a board then runs with plain ``--target <board>``
and no explicit paths.  ``prepare.py`` reports the same digest comparison
for self-compiled pairs.

Usage (from the repository root)::

    python samples/vision/fasterrcnn/model/download.py                  # all targets
    python samples/vision/fasterrcnn/model/download.py --target s100    # one target
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from samples.vision.fasterrcnn.runtime.python.cli import (  # noqa: E402
    MODEL_DIR,
    SUPPORTED_TARGETS,
    descriptor_pairs,
)
from utils.py_utils.assets import download_asset, list_assets  # noqa: E402

MANIFEST_GROUP = {"x5": "x5", "s100": "s", "s100p": "s", "s600": "s"}


def manifest_filename(target: str, filename: str) -> str:
    """Return the manifest row filename for one artifact of one target."""
    return filename if target == "x5" else f"{target}/{filename}"


def pair_assets(target: str, pairs):
    """Resolve one target's manifest asset rows, verified against the descriptor."""
    pair = pairs[target]
    group = MANIFEST_GROUP[target]
    assets = {asset.filename: asset for asset in list_assets(group, "fasterrcnn")}
    resolved = []
    for role, filename, expected in (
            ("parta", pair.parta_file, pair.parta_sha256),
            ("partb", pair.partb_file, pair.partb_sha256)):
        key = manifest_filename(target, filename)
        if key not in assets:
            raise ValueError(
                f"{target}/{role}: manifest {group} has no fasterrcnn asset "
                f"{key!r}")
        asset = assets[key]
        if asset.sha256 is None or asset.sha256.lower() != expected.lower():
            raise ValueError(
                f"{target}/{role}: manifest digest for {key} does not match "
                f"the descriptor (manifest "
                f"{asset.sha256!r}, descriptor {expected!r})")
        resolved.append(asset)
    return pair, resolved


def download_target(target: str, output_dir: Path, pairs) -> list[Path]:
    """Fetch one target's pair into ``output_dir/<target>/``; return the paths."""
    pair, assets = pair_assets(target, pairs)
    destinations = []
    for asset, default in zip(assets, (pair.parta_path, pair.partb_path)):
        destination = output_dir / default.relative_to(MODEL_DIR)
        digest = download_asset(asset, destination)
        print(f"{target}: {destination.name} sha256 {digest} -> {destination}")
        destinations.append(destination)
    return destinations


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="download",
        description="Download published FasterRCNN PartA/PartB pairs "
                    "into model/<target>/ (default: all four targets)")
    parser.add_argument("--target", nargs="+", choices=SUPPORTED_TARGETS,
                        default=list(SUPPORTED_TARGETS),
                        help="board target(s) to fetch (default: all)")
    parser.add_argument("--output-dir", type=Path, default=MODEL_DIR,
                        help="destination directory (default: the sample "
                             "model directory, the runtime's lookup path)")
    args = parser.parse_args(argv)

    try:
        pairs = descriptor_pairs()
        for target in args.target:
            download_target(target, args.output_dir, pairs)
    except (ImportError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
