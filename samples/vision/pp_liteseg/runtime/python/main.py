# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""People and Agent entrypoint for the PP-LiteSeg segmentation sample.

This file stays deliberately small: parse the arguments, resolve the model,
construct the segmenter, call ``predict``, present results. Option
declarations, model-free listing/dry-run, and artifact writing live in
``cli.py``; the segmentation flow lives in ``pp_liteseg.py``.
"""

from __future__ import annotations

from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[5]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from samples.vision.pp_liteseg.runtime.python.cli import (  # noqa: E402
    build_parser, class_names_for, read_image, resolve_selection,
    run_dry_run, run_list_models, runtime_report, save_results,
)


def main(argv=None) -> int:
    """Run the requested PP-LiteSeg command and return its exit status.

    Args:
        argv: Optional command-line argument sequence, excluding the program
            name. None reads sys.argv through argparse.

    Returns:
        int: 0 for success; 2 for a reported selection, IO, or runtime error.

    Raises:
        SystemExit: argparse handles --help or rejects invalid arguments.

    Notes:
        List and dry-run modes do not load a model or the SDK. Inference
        writes the rendered image, class-mask NPY, and JSON report artifacts.
    """
    args = build_parser().parse_args(argv)
    if args.list_models:
        return run_list_models(args.target)
    try:
        selection = resolve_selection(args.target, asset_id=args.asset_id, model_path=args.model_path)
        if (args.input_width, args.input_height) != (1024, 512):
            raise ValueError('The published model requires input-width=1024 and input-height=512')
        if not 0 <= args.alpha <= 1:
            raise ValueError('alpha must be in [0,1]')
        if args.priority is not None and not 0 <= args.priority <= 255:
            raise ValueError('priority must be 0..255')
        if args.bpu_cores is not None and any(i < 0 for i in args.bpu_cores):
            raise ValueError('bpu-cores must be nonnegative')
        if args.mask_save_path.suffix != '.npy':
            raise ValueError('mask-save-path must end in .npy')
        if args.dry_run:
            return run_dry_run(selection, args.test_img)

        # Real execution starts here: OpenCV and the SDK load only after the
        # selection and option checks above have passed.
        from samples.vision.pp_liteseg.runtime.python.pp_liteseg import PPLiteSegSegmenter
        from samples.vision.pp_liteseg.runtime.python.cli import render_result

        model = PPLiteSegSegmenter(selection)
        model.set_scheduling_params(priority=args.priority, bpu_cores=args.bpu_cores)
        image = read_image(args.test_img)
        labels = model.predict(image)
        result = render_result(image, labels, alpha=args.alpha)
        report = runtime_report(selection, model.binding,
                                str(getattr(model.runner.runtime, 'version', 'unknown')),
                                args.test_img, labels, result.shape,
                                args.output, args.mask_save_path)
        report['class_names'] = class_names_for(report['class_ids'])
        save_results(args.output, args.mask_save_path, args.report_path,
                     result, labels, report)
        print(json.dumps(report, indent=2))
        return 0
    except (ValueError, OSError, RuntimeError, ImportError) as exc:
        print(f'error: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
