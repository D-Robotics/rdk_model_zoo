# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Explicit SigLIP download; shared S100/S100P publication facts stay in manifest."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
from samples._shared.assets import download_asset
from samples.vision.siglip.runtime.python.model_binding import DEFAULT_VARIANT, VARIANTS, SUPPORTED_TARGETS, resolve_selection

DEFAULT_OUTPUT_DIR=Path(__file__).resolve().parent


def download_target(target='s100',output_dir=None,*,variant=DEFAULT_VARIANT):
    """Fetch one exact HBM into its published relative path and print digest.

    S100P intentionally uses the same s100/ artifact because the fixed source
    supports Nash-E and Nash-M with those bytes. This is not target fallback.
    """
    selection=resolve_selection(target,variant=variant)
    destination=(Path(output_dir).expanduser() if output_dir is not None else DEFAULT_OUTPUT_DIR)/selection.asset.filename
    digest=download_asset(selection.asset,destination)
    print(f'Downloaded {selection.asset.reference} to {destination}')
    print(f'Observed SHA-256: {digest}')
    if selection.asset.sha256 is None:
        print('Publisher SHA-256 is unknown; observed digest does not independently verify origin.')
    return digest


def build_parser():
    """SDK-free explicit preparation parser."""
    p=argparse.ArgumentParser(description='Download one published SigLIP packed HBM.')
    p.add_argument('--target',choices=SUPPORTED_TARGETS,default='s100')
    p.add_argument('--variant',choices=tuple(VARIANTS),default=DEFAULT_VARIANT)
    p.add_argument('--output-dir',type=Path,default=DEFAULT_OUTPUT_DIR)
    return p


def main(argv=None):
    args=build_parser().parse_args(argv)
    try:
        download_target(args.target,args.output_dir,variant=args.variant)
    except (OSError,ValueError) as exc:
        print(f'error: {exc}',file=sys.stderr)
        return 2
    return 0


if __name__=='__main__':
    raise SystemExit(main())
