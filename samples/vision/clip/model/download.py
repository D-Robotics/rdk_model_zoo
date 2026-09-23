# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Explicit paired CLIP model download through the repository asset manifest."""
import argparse
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from samples._shared.assets import download_asset
from samples.vision.clip.runtime.python.model_binding import list_available_assets

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent


def download_target(target='x5', output_dir=None):
    """Prepare both required encoders and report each observed SHA-256."""
    if target != 'x5':
        raise ValueError(f'No published CLIP pair for {target}.')
    directory = Path(output_dir).expanduser() if output_dir is not None else DEFAULT_OUTPUT_DIR
    digests = {}
    for asset in list_available_assets(target):
        destination = directory / asset.filename
        digest = download_asset(asset, destination)
        digests[asset.reference] = digest
        print(f'Downloaded {asset.reference} to {destination}\nObserved SHA-256: {digest}')
        if asset.sha256 is None:
            print('Publisher SHA-256 is unknown; observed digest does not independently verify origin.')
    return digests


def build_parser():
    parser = argparse.ArgumentParser(description='Download both CLIP X5 encoders.')
    parser.add_argument('--target', choices=('x5',), default='x5')
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        download_target(args.target, args.output_dir)
    except (OSError, ValueError) as exc:
        print(f'error: {exc}', file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
