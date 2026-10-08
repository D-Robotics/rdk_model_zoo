# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Depth CLI entry: construct the task, run one ``predict``, archive evidence.

Option declarations, the model-free listing/dry-run modes and the canonical
output/report writing live in ``cli.py``.  This entry stays focused on the
execution path: resolve the selection, load the runner, construct
``DepthAnythingV2Task`` and call ``predict`` once.  The canonical output
contract includes ``raw_depth.npy``, so the entry requests this call's raw
tensor through the opt-in ``return_details`` record instead of recomputing
stages.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from samples.vision.depth_anything_v2.runtime.python.cli import (
    build_parser,  # re-exported here: the contract checker imports it from main
    read_bgr_image,
    run_dry_run,
    run_list_models,
    save_depth_evidence,
    validate_output_paths,
)
from samples.vision.depth_anything_v2.runtime.python.cli import (
    resolve_selection,
)



def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        if args.list_models:
            return run_list_models(args.target)
        selection = resolve_selection(
            args.target, asset_id=args.asset_id, model_path=args.model_path
        )
        if not 0 <= args.priority <= 255 or any(core < 0 for core in args.bpu_cores):
            raise ValueError("Priority must be 0..255 and BPU cores nonnegative")
        if args.dry_run:
            return run_dry_run(selection, args)
        output = args.output.expanduser()
        extra = args.img_save_path.expanduser() if args.img_save_path else None
        validate_output_paths(output, extra)
        image_path = args.test_img.expanduser()
        image = read_bgr_image(image_path)

        # Real execution starts here: construction gates board identity and
        # the publication hash before the SDK import.
        from samples.vision.depth_anything_v2.runtime.python.depth_anything_v2 import (
            DepthEstimator,
        )

        task = DepthEstimator(selection, resize_type=args.resize_type)
        task.set_scheduling_params(priority=args.priority, bpu_cores=args.bpu_cores)
        details = task.predict(image, return_details=True)
        save_depth_evidence(
            output,
            extra,
            selection=selection,
            runner=task.runner,
            binding=task.binding,
            args=args,
            image_path=image_path,
            details=details,
        )
        print(f"Saved relative depth and provenance to {output}")
        return 0
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
