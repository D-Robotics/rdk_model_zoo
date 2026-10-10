# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Explicitly download the sole published KWS S100 model."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from utils.py_utils.assets import download_asset
from samples.speech.kws.runtime.python.cli import ASSET_ID, resolve_selection


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=("s100",), default="s100")
    parser.add_argument("--asset-id", default=ASSET_ID)
    parser.add_argument(
        "--output-dir", type=Path, default=Path(__file__).resolve().parent
    )
    args = parser.parse_args(argv)
    try:
        selection = resolve_selection(args.target, asset_id=args.asset_id)
        path = args.output_dir.expanduser() / selection.asset.filename
        digest = download_asset(selection.asset, path)
        print(f"{selection.asset.reference}\nPath: {path}\nObserved SHA-256: {digest}")
        print(
            "Publisher SHA-256 is not recorded; this digest binds bytes, not independently authenticated origin."
        )
        return 0
    except (ValueError, OSError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
