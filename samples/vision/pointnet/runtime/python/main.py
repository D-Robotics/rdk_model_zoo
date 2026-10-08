# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""PointNet CLI entry: construct the task, run one ``predict``, save the evidence.

Option declarations, the model-free listing/dry-run modes and the
label/plot/report writing live in ``cli.py``.  This entry stays focused on the
execution path: resolve the selection, load the raw points and runner,
construct ``PointNetTask`` and call ``predict`` once.  The report and plots
need this call's normalized cloud and centroid/radius context, so the entry
requests them through the opt-in ``return_details`` record instead of
recomputing stages.
"""
from __future__ import annotations
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[5]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from samples.vision.pointnet.runtime.python.cli import (
    build_parser,  # re-exported here: existing callers import it from main
    run_dry_run,
    run_list_models,
    save_pointnet_evidence,
)
from samples.vision.pointnet.runtime.python.cli import resolve_selection


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.list_models:
            return run_list_models(args.target)
        selection = resolve_selection(args.target, asset_id=args.asset_id, model_path=args.model_path)
        if not 0 <= args.priority <= 255 or any(x < 0 for x in args.bpu_cores):
            raise ValueError('priority must be 0..255 and bpu-cores must be nonnegative')
        if args.dry_run:
            return run_dry_run(selection)
        import numpy as np
        from samples.vision.pointnet.runtime.python.pointnet import PointNetSegmenter

        input_path = args.test_pts.expanduser()
        points = np.loadtxt(input_path, dtype=np.float32, ndmin=2)
        model = PointNetSegmenter(selection)
        model.set_scheduling_params(priority=args.priority, bpu_cores=args.bpu_cores)
        details = model.predict(points, return_details=True)
        save_pointnet_evidence(args.output_dir.expanduser(),
                               selection=selection, binding=model.binding,
                               input_path=args.test_pts, details=details,
                               no_plot=args.no_plot)
        return 0
    except (ValueError, OSError, RuntimeError, ImportError) as exc:
        print(f'error: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
