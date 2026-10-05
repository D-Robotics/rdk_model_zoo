# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Depth CLI entry: construct the task, run one measured ``predict``, save evidence.

Option declarations, the model-free listing/dry-run modes and the canonical
output/report writing live in ``cli.py``.  This entry stays focused on the
execution path: resolve the selection, load the runner, construct
``Yolo26DepthTask`` and call ``predict`` once.  The report needs the explicit
warmup count and the one-forward latency, so the entry requests them through
the opt-in ``return_details`` record; the timed forward excludes
preprocessing/postprocessing and no second inference chain is run.
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[5]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from samples.vision.yolo26_depth.runtime.python.cli import (
    build_parser,  # re-exported here: the contract checker imports it from main
    read_bgr_image,
    run_dry_run,
    run_list_models,
    save_depth_evidence,
)
from samples.vision.yolo26_depth.runtime.python.model_binding import (
    resolve_selection,
)


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        if args.list_models:
            return run_list_models(args.target)
        selection = resolve_selection(
            args.target,
            variant=args.variant,
            asset_id=args.asset_id,
            model_path=args.model_path,
            converted_model=args.converted_model,
        )
        if args.warmup < 0:
            raise ValueError("warmup must be nonnegative")
        if args.priority is not None and not 0 <= args.priority <= 255:
            raise ValueError("priority must be 0..255")
        if args.bpu_cores is not None and any(i < 0 for i in args.bpu_cores):
            raise ValueError("bpu-cores must be nonnegative")
        if args.dry_run:
            return run_dry_run(selection, args)
        output = args.output.expanduser()
        if output.exists():
            raise FileExistsError(f"Use a new output directory: {output}")
        from samples.vision.yolo26_depth.runtime.python.model_runner import (
            RuntimeModelRunner,
        )
        from samples.vision.yolo26_depth.runtime.python.yolo26_depth import (
            Yolo26DepthTask,
        )

        image_path = args.test_img.expanduser()
        image = read_bgr_image(image_path)
        runner = RuntimeModelRunner(selection)
        binding = runner.load()
        priority = (
            args.priority
            if args.priority is not None
            else (None if selection.target == "x5" else 0)
        )
        cores = (
            args.bpu_cores
            if args.bpu_cores is not None
            else (None if selection.target == "x5" else [0])
        )
        runner.set_scheduling_params(priority=priority, bpu_cores=cores)
        task = Yolo26DepthTask(runner, binding)
        details = task.predict(
            image, warmup=args.warmup, return_details=True
        )
        save_depth_evidence(
            output,
            selection=selection,
            runner=runner,
            binding=binding,
            image_path=image_path,
            image=image,
            details=details,
            priority=priority,
            bpu_cores=cores,
        )
        return 0
    except (ValueError, OSError, RuntimeError, ImportError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
