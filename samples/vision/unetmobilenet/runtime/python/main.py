# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""People and Agent entrypoint for the UnetMobileNet segmentation sample.

This file stays deliberately small: parse the arguments, resolve the model,
construct the segmenter, call ``predict``, present results. Option
declarations, model-free listing/dry-run, and artifact writing live in
``cli.py``; the segmentation flow lives in ``unetmobilenet.py``.
"""

from __future__ import annotations

from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[5]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from samples.vision.unetmobilenet.runtime.python.cli import (  # noqa: E402
    build_parser, read_image, resolve_selection, run_dry_run, run_list_models,
    runtime_report, save_results,
)


def main(argv=None) -> int:
    """Run the requested UnetMobileNet command and return its exit status.

    Args:
        argv: Optional command-line argument sequence, excluding the program
            name. None reads sys.argv through argparse.

    Returns:
        int: 0 for success; 2 for a reported selection, IO, or runtime error.

    Raises:
        SystemExit: argparse handles --help or rejects invalid arguments.

    Notes:
        List and dry-run modes do not load a model or the SDK. Inference
        writes the overlay image, class-mask NPY, and JSON report artifacts.
    """
    args = build_parser().parse_args(argv)
    if args.list_models:
        return run_list_models(args.target)
    try:
        selection = resolve_selection(args.target, asset_id=args.asset_id, model_path=args.model_path)
        if not 0 <= args.alpha_f <= 1:
            raise ValueError('alpha-f must be in [0,1]')
        if not 0 <= args.priority <= 255 or any(core < 0 for core in args.bpu_cores):
            raise ValueError('priority must be 0..255 and bpu-cores must be nonnegative')
        if args.mask_save_path.suffix != '.npy':
            raise ValueError('mask-save-path must end in .npy')
        if args.dry_run:
            return run_dry_run(selection, args.test_img)

        # Real execution starts here: OpenCV and the SDK load only after the
        # selection and option checks above have passed.
        from samples.vision.unetmobilenet.runtime.python.unetmobilenet import UnetMobileNetSegmenter
        from samples.vision.unetmobilenet.runtime.python.cli import render_overlay

        model = UnetMobileNetSegmenter(selection)
        model.set_scheduling_params(priority=args.priority, bpu_cores=args.bpu_cores)
        image = read_image(args.test_img)
        labels = model.predict(image)
        overlay = render_overlay(image, labels, alpha_f=args.alpha_f)
        report = runtime_report(selection, model.binding,
                                str(getattr(model.runner.runtime, 'version', 'unknown')),
                                args.test_img, labels, args.alpha_f,
                                args.img_save_path, args.mask_save_path)
        save_results(args.img_save_path, args.mask_save_path, args.report_path,
                     overlay, labels, report)
        print(json.dumps(report, indent=2))
        return 0
    except (ValueError, RuntimeError, OSError, ImportError) as exc:
        print(f'error: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
