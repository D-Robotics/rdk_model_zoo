"""Explicitly prepare the manifest YOLOWorld model; never called by runtime."""
from __future__ import annotations
import argparse, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from samples._shared.assets import download_asset
from samples.vision.yoloworld.runtime.python.model_binding import list_available_assets
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent

def download_target(target='x5', output_dir=None):
    if target != 'x5': raise ValueError(f'No published YOLOWorld model for {target}.')
    directory = Path(output_dir).expanduser() if output_dir is not None else DEFAULT_OUTPUT_DIR
    asset = list_available_assets(target)[0]; destination = directory / asset.filename
    digest = download_asset(asset, destination)
    print(f'Downloaded {asset.reference} to {destination}\nObserved SHA-256: {digest}')
    if asset.sha256 is None: print('Publisher SHA-256 is unknown; observed digest does not verify origin.')
    return {asset.reference: digest}

def build_parser():
    p = argparse.ArgumentParser(description='Download the YOLOWorld X5 model explicitly.')
    p.add_argument('--target', choices=('x5',), default='x5'); p.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR); return p

def main(argv=None):
    try: download_target(**vars(build_parser().parse_args(argv))); return 0
    except (OSError, ValueError, RuntimeError) as exc: print(f'error: {exc}', file=sys.stderr); return 2
if __name__ == '__main__': raise SystemExit(main())
