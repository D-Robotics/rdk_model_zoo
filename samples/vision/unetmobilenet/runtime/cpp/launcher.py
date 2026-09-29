# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Explicit preparation/identity gate before native build or execution."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[5]
CPP = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from samples._shared.assets import verify_asset_file
from samples._shared.platforms import require_execution_target
from samples.vision.unetmobilenet.runtime.python.model_binding import (
    SAMPLE_DIR, SUPPORTED_TARGETS, resolve_selection, list_available_assets,
)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target', choices=('auto',)+SUPPORTED_TARGETS, default='auto')
    parser.add_argument('--asset-id')
    parser.add_argument('--model-path', type=Path)
    parser.add_argument('--test-img', type=Path, default=SAMPLE_DIR/'test_data/segmentation.png')
    parser.add_argument('--img-save-path', type=Path, default=Path('result.jpg'))
    parser.add_argument('--mask-save-path', type=Path, default=Path('unetmobilenet_mask.png'))
    parser.add_argument('--report-path', type=Path, default=Path('unetmobilenet_cpp_report.json'))
    parser.add_argument('--alpha-f', type=float, default=0.75)
    parser.add_argument('--priority', type=int, default=0)
    parser.add_argument('--bpu-core', type=int, default=-1)
    parser.add_argument('--binary', type=Path)
    parser.add_argument('--build', action='store_true')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--dry-run', action='store_true')
    mode.add_argument('--list-models', action='store_true')
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        if args.list_models:
            print(json.dumps([a.reference for a in list_available_assets(args.target)], indent=2))
            return 0
        selection = resolve_selection(args.target, asset_id=args.asset_id, model_path=args.model_path)
        if not 0 <= args.alpha_f <= 1 or not 0 <= args.priority <= 255 or not -1 <= args.bpu_core <= 3:
            raise ValueError('alpha-f 0..1, priority 0..255, bpu-core -1 or 0..3 required')
        if args.mask_save_path.suffix != '.png':
            raise ValueError('Native mask-save-path must end in .png')
        build_dir = CPP/'build'/selection.target
        binary = args.binary.expanduser().resolve() if args.binary else build_dir/'unetmobilenet_cpp'
        if args.build and args.binary:
            raise ValueError('--build uses the target build directory; omit --binary')
        command = [str(binary), '--target', selection.target, '--model-path', str(selection.model_path),
                   '--test-img', str(args.test_img.expanduser()), '--alpha-f', str(args.alpha_f),
                   '--img-save-path', str(args.img_save_path.expanduser()),
                   '--mask-save-path', str(args.mask_save_path.expanduser()),
                   '--report-path', str(args.report_path.expanduser()),
                   '--priority', str(args.priority), '--bpu-core', str(args.bpu_core)]
        if args.dry_run:
            print(json.dumps({'target':selection.target, 'asset_id':selection.asset.reference,
                              'command':command, 'build_directory':str(build_dir),
                              'executed':False, 'downloaded':False}, indent=2))
            return 0
        require_execution_target(selection.target)
        verify_asset_file(selection.asset, selection.model_path)
        if not args.test_img.expanduser().is_file():
            raise ValueError('Input image does not exist')
        if args.build:
            subprocess.run(['cmake', '-S', str(CPP), '-B', str(build_dir),
                            f'-DUNETMOBILENET_TARGET={selection.target}', '-DCMAKE_BUILD_TYPE=Release'], check=True)
            subprocess.run(['cmake', '--build', str(build_dir), '--parallel', '2'], check=True)
        if not binary.is_file():
            raise ValueError('Native binary missing; build explicitly with --build on the target board')
        return subprocess.run(command, check=False).returncode
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        print(f'error: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
