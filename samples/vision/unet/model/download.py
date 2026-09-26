# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Manifest-backed download of one or more UNet X5 variants."""
import argparse
from pathlib import Path
from samples._shared.assets import download_asset
from samples.vision.unet.runtime.python.model_binding import VARIANTS, resolve_selection


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--target',choices=('x5',),default='x5')
    p.add_argument('--variant',choices=VARIANTS+('all',),nargs='+',default=['resnet18'])
    p.add_argument('--output-dir',type=Path,default=Path(__file__).resolve().parent)
    args=p.parse_args(argv)
    variants=VARIANTS if 'all' in args.variant else dict.fromkeys(args.variant)
    for variant in variants:
        s=resolve_selection(args.target,variant=variant)
        path=args.output_dir/s.asset.filename
        digest=download_asset(s.asset,path)
        print(f'Prepared {s.asset.reference} at {path}; verified_sha256={digest}')
    return 0


if __name__=='__main__':
    raise SystemExit(main())
