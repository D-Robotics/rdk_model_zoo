# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""YOLOE CLI entry: construct the model, run one ``predict``, show the result.

Option declarations, the model-free listing/dry-run modes and the result
report live in ``cli.py``.  This entry stays focused on the execution path:
resolve the selection, build the config, construct ``YOLOE`` (it loads the
selected artifact itself) and call ``predict`` once; rendering the
annotated image is a separate visualization step.
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[5]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from samples.vision.yoloe.runtime.python.cli import (
    build_parser,  # re-exported here: the contract checker imports it from main
    print_result_report,
    run_dry_run,
    run_list_models,
    validate_scheduling,
)
from samples.vision.yoloe.runtime.python.model_binding import (
    resolve_selection,
)


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        if args.list_models:
            return run_list_models(args.target)
        if args.dry_run and args.target == "auto":
            raise ValueError("Host dry-run requires an explicit --target.")
        selection = resolve_selection(
            args.target,
            variant=args.variant,
            asset_id=args.asset_id,
            model_path=args.model_path,
            local_float_sha256=args.local_float_sha256,
        )
        from samples.vision.yoloe.runtime.python.config import Config, validate_config

        config = Config(
            args.score_thres,
            args.nms_thres,
            args.resize_type,
            selection.target != "x5"
            and selection.variant.startswith("11")
            and not args.no_morph,
            args.max_det,
            not args.multi_label,
        )
        validate_config(selection, config)
        validate_scheduling(args)
        if args.dry_run:
            return run_dry_run(selection, config)
        from samples.vision.yoloe.runtime.python.yoloe import YOLOE
        from samples.vision.yoloe.runtime.python.visualization import (
            load_inputs,
            save_result,
        )

        image, labels = load_inputs(args.test_img, args.label_file)
        model = YOLOE(selection, config)
        model.set_scheduling_params(priority=args.priority, bpu_cores=args.bpu_cores)
        result = model.predict(image)
        save_result(
            args.img_save_path, image, result, labels, contours=not args.no_contour
        )
        print_result_report(result, selection, args)
        return 0
    except (ValueError, TypeError, OSError, RuntimeError, ImportError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
