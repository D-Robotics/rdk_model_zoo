#!/usr/bin/env python3
# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Single-image compatibility entry; delegates all inference to the runtime."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from samples.vision.pp_liteseg.runtime.python.main import main as runtime_main
from samples.vision.pp_liteseg.runtime.python.model_binding import ASSET_ID


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--image', required=True)
    parser.add_argument('--output', default='result.jpg')
    parser.add_argument('--alpha', type=float, default=0.55)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    output = Path(args.output)
    return runtime_main([
        '--target', 'x5', '--asset-id', ASSET_ID, '--model-path', args.model,
        '--test-img', args.image, '--output', str(output), '--alpha', str(args.alpha),
        '--mask-save-path', str(output.with_suffix('.labels.npy')),
        '--report-path', str(output.with_suffix('.report.json')),
    ])


if __name__ == '__main__':
    raise SystemExit(main())
