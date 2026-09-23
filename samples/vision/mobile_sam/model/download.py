"""Prepare MobileSAM's exact manifest-backed encoder/decoder pair."""
from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from samples._shared.assets import download_asset  # noqa: E402

binding = importlib.import_module("samples.vision.mobile_sam.runtime.python.model_binding")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare MobileSAM encoder and decoder assets.")
    parser.add_argument("--target", choices=("x5", "s100", "s100p", "s600"), required=True)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    return parser


def download_target(target: str, output_dir: str | Path) -> tuple[str, ...]:
    assets = binding.list_available_assets(target)
    if len(assets) != 2:
        raise ValueError(f"Expected an encoder/decoder pair for {target}, found {len(assets)}.")
    output = Path(output_dir).expanduser()
    digests = []
    for asset in assets:
        destination = output / asset.filename
        digest = download_asset(asset, destination)
        digests.append(digest)
        publisher = asset.sha256 or "unknown"
        print(f"Prepared {asset.reference} at {destination}; observed_sha256={digest}; publisher_sha256={publisher}")
    return tuple(digests)


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        download_target(args.target, args.output_dir)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
