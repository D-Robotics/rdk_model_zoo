# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""UNetMobileNet CLI: IO, rendering and metadata reports outside task stages."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[5]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from samples.vision.unetmobilenet.runtime.python.model_binding import (
    SAMPLE_DIR, SUPPORTED_TARGETS, list_available_assets, resolve_selection,
)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target', choices=('auto',)+SUPPORTED_TARGETS, default='auto')
    parser.add_argument('--asset-id')
    parser.add_argument('--model-path', type=Path)
    parser.add_argument('--test-img', type=Path, default=SAMPLE_DIR/'test_data/segmentation.png')
    parser.add_argument('--img-save-path', type=Path, default=Path('result.jpg'))
    parser.add_argument('--mask-save-path', type=Path, default=Path('unetmobilenet_mask.npy'))
    parser.add_argument('--report-path', type=Path, default=Path('unetmobilenet_report.json'))
    parser.add_argument('--alpha-f', type=float, default=0.75, help='Weight of ORIGINAL image, 0..1')
    parser.add_argument('--priority', type=int, default=0)
    parser.add_argument('--bpu-cores', nargs='+', type=int, default=[0])
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--list-models', action='store_true')
    mode.add_argument('--dry-run', action='store_true')
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        if args.list_models:
            print(json.dumps([{'asset_id':a.reference, 'url':a.url, 'sha256':a.sha256}
                              for a in list_available_assets(args.target)], indent=2))
            return 0
        selection = resolve_selection(args.target, asset_id=args.asset_id, model_path=args.model_path)
        if not 0 <= args.alpha_f <= 1:
            raise ValueError('alpha-f must be in [0,1]')
        if not 0 <= args.priority <= 255 or any(core < 0 for core in args.bpu_cores):
            raise ValueError('priority must be 0..255 and bpu-cores must be nonnegative')
        if args.mask_save_path.suffix != '.npy':
            raise ValueError('mask-save-path must end in .npy')
        if args.dry_run:
            print(json.dumps({'target':selection.target, 'asset_id':selection.asset.reference,
                              'model_path':str(selection.model_path), 'test_img':str(args.test_img),
                              'input_shapes':[[1,1024,2048,1],[1,512,1024,2]],
                              'output':'original-resolution int32 class IDs 0..18',
                              'sdk_loaded':False, 'downloaded':False}, indent=2))
            return 0
        import cv2
        import numpy as np
        from samples._shared.runtime_meta import metadata_evidence
        from samples.vision.unetmobilenet.runtime.python.model_runner import RuntimeModelRunner
        from samples.vision.unetmobilenet.runtime.python.unetmobilenet import UnetMobileNetTask
        from samples.vision.unetmobilenet.runtime.python.visualization import render_overlay
        runner = RuntimeModelRunner(selection)
        binding = runner.load()
        runner.set_scheduling_params(priority=args.priority, bpu_cores=args.bpu_cores)
        image = cv2.imread(str(args.test_img.expanduser()), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f'Cannot read image: {args.test_img}')
        labels = UnetMobileNetTask(runner, binding).predict(image)
        overlay = render_overlay(image, labels, alpha_f=args.alpha_f)
        image_path, mask_path, report_path = [path.expanduser() for path in (
            args.img_save_path, args.mask_save_path, args.report_path)]
        for path in (image_path, mask_path, report_path):
            path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(image_path), overlay):
            raise OSError(f'Cannot write {image_path}')
        np.save(mask_path, labels, allow_pickle=False)
        report = {
            'target':selection.target, 'asset_id':selection.asset.reference,
            'model_path':str(selection.model_path), 'input_path':str(args.test_img),
            'publisher_sha256':selection.asset.sha256,
            'runtime_version':str(getattr(runner.runtime, 'version', 'unknown')),
            'metadata':metadata_evidence(binding.metadata),
            'mask_shape':list(labels.shape), 'class_ids':np.unique(labels).tolist(),
            'alpha_f':args.alpha_f, 'img_save_path':str(image_path), 'mask_save_path':str(mask_path),
        }
        text = json.dumps(report, indent=2)
        report_path.write_text(text+'\n')
        print(text)
        return 0
    except (ValueError, RuntimeError, OSError, ImportError) as exc:
        print(f'error: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
