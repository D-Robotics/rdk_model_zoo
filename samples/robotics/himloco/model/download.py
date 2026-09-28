"""Explicit published HIMLoco model preparation; inference never downloads."""

import argparse
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from samples._shared.assets import download_asset
from samples.robotics.himloco.runtime.python.model_binding import resolve_selection


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=["x5"], default="x5")
    parser.add_argument(
        "--output-dir", type=Path, default=Path(__file__).resolve().parent
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        asset = resolve_selection(args.target).asset
        path = args.output_dir.expanduser().resolve() / "bayes-e" / asset.filename
        record = {
            "asset_id": asset.reference,
            "url": asset.url,
            "path": str(path),
            "expected_sha256": asset.sha256,
        }
        if not args.dry_run:
            record["observed_sha256"] = download_asset(asset, path)
            record["prepared"] = True
        print(json.dumps(record, indent=2))
        return 0
    except Exception as error:
        print(f"Model preparation failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
