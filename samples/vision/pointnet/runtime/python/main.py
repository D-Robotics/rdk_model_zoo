# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""PointNet CLI: file IO and optional plotting live outside inference stages."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[5]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from samples.vision.pointnet.runtime.python.model_binding import (
    SAMPLE_DIR, SUPPORTED_TARGETS, list_available_assets, resolve_selection,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='PointNet chair part segmentation')
    parser.add_argument('--target', choices=('auto',) + SUPPORTED_TARGETS, default='auto')
    parser.add_argument('--asset-id')
    parser.add_argument('--model-path', help='External HBM path; requires exact --asset-id')
    parser.add_argument('--test-pts', type=Path, default=SAMPLE_DIR/'test_data/chair.pts')
    parser.add_argument('--output-dir', type=Path, default=Path('outputs/pointnet'))
    parser.add_argument('--no-plot', action='store_true', help='Save labels/report without matplotlib images')
    parser.add_argument('--priority', type=int, default=0)
    parser.add_argument('--bpu-cores', nargs='+', type=int, default=[0])
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--list-models', action='store_true')
    modes.add_argument('--dry-run', action='store_true')
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.list_models:
            print(json.dumps([{'asset_id': a.reference, 'target': 's100', 'url': a.url,
                               'sha256': a.sha256} for a in list_available_assets(args.target)], indent=2))
            return 0
        selection = resolve_selection(args.target, asset_id=args.asset_id, model_path=args.model_path)
        if not 0 <= args.priority <= 255 or any(x < 0 for x in args.bpu_cores):
            raise ValueError('priority must be 0..255 and bpu-cores must be nonnegative')
        if args.dry_run:
            print(json.dumps({'target': selection.target, 'asset_id': selection.asset.reference,
                              'model_path': str(selection.model_path), 'input': '(1,3,N) float32',
                              'output': '(1,N,4) logits; N and raw dtype checked at load',
                              'sdk_loaded': False, 'downloaded': False,
                              'model_path_exists': selection.model_path.is_file()}, indent=2))
            return 0
        import numpy as np
        from samples.vision.pointnet.runtime.python.model_runner import RuntimeModelRunner
        from samples.vision.pointnet.runtime.python.pointnet import PointNetTask
        from samples._shared.runtime_meta import metadata_evidence
        points = np.loadtxt(args.test_pts.expanduser(), dtype=np.float32, ndmin=2)
        runner = RuntimeModelRunner(selection)
        binding = runner.load()
        runner.set_scheduling_params(priority=args.priority, bpu_cores=args.bpu_cores)
        task = PointNetTask(runner,binding)
        prepared = task.pre_process(points)
        labels = task.post_process(task.forward(prepared.tensors))
        out = args.output_dir.expanduser()
        out.mkdir(parents=True, exist_ok=True)
        np.save(out/'labels.npy',labels,allow_pickle=False)
        if not args.no_plot:
            from samples.vision.pointnet.runtime.python.visualization import save_original_view, save_segmentation_view
            normalized = prepared.tensors[binding.input_name][0].T
            save_original_view(normalized,str(out/'result_orig.png'))
            save_segmentation_view(normalized,labels,str(out/'result.png'))
        counts = {name:int(np.count_nonzero(labels==i)) for i,name in enumerate(('back','seat','leg','arm'))}
        report = {'target': selection.target, 'asset_id': selection.asset.reference,
                'input': str(args.test_pts), 'point_count': len(labels), 'counts': counts,
                'normalization': {'centroid': prepared.context.centroid, 'radius': prepared.context.radius},
                'metadata': metadata_evidence(binding.metadata), 'output_dir': str(out)}
        text = json.dumps(report,indent=2)
        (out / 'result.json').write_text(text + '\n')
        print(text)
        return 0
    except (ValueError, OSError, RuntimeError, ImportError) as exc:
        print(f'error: {exc}',file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
