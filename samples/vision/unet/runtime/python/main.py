# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""People and Agent entrypoint for the UNet segmentation sample.

This file stays deliberately small: parse the arguments, resolve the model,
construct the segmenter, call ``predict``, present results. Option
declarations, model-free listing/dry-run, and artifact writing live in
``cli.py``; the segmentation flow lives in ``unet.py``.
"""

from __future__ import annotations

from pathlib import Path
import json
import sys
import time

ROOT = Path(__file__).resolve().parents[5]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from samples.vision.unet.runtime.python.cli import (  # noqa: E402
    BindingError, build_parser, read_image, resolve_selection,
    run_dry_run, run_list_models, runtime_report, save_results,
)


def main(argv=None) -> int:
    """Run the requested UNet command and return its exit status.

    Args:
        argv: Optional command-line argument sequence, excluding the program
            name. None reads sys.argv through argparse.

    Returns:
        int: 0 for success; 2 for a reported selection, IO, or runtime error.

    Raises:
        SystemExit: argparse handles --help or rejects invalid arguments.

    Notes:
        List and dry-run modes do not load a model or the SDK. Inference
        writes the mask, overlay, and JSON report artifacts.
    """
    args = build_parser().parse_args(argv)
    if args.list_models:
        return run_list_models(args.target)
    try:
        selection = resolve_selection(args.target, variant=args.variant,
                                      asset_id=args.asset_id, model_path=args.model_path)
        if not 0 <= args.alpha <= 1:
            raise ValueError('alpha must be between 0 and 1')
        if args.priority is not None and not 0 <= args.priority <= 255:
            raise ValueError('priority must be 0..255')
        if args.bpu_core is not None and args.bpu_core < 0:
            raise ValueError('bpu-core must be nonnegative')
        if args.dry_run:
            return run_dry_run(selection)

        # Real execution starts here: OpenCV and the SDK load only after the
        # selection and option checks above have passed.
        import cv2
        from samples.vision.unet.runtime.python.unet import UNetSegmenter
        from samples.vision.unet.runtime.python.cli import colorize_mask

        model = UNetSegmenter(selection)
        model.set_scheduling_params(priority=args.priority,
                                     bpu_cores=[args.bpu_core] if args.bpu_core is not None else None)
        image = read_image(args.test_img)
        start = time.perf_counter()
        mask = model.predict(image)
        elapsed = (time.perf_counter() - start) * 1000
        resized = cv2.resize(image, (512, 512), interpolation=cv2.INTER_LINEAR)
        overlay = cv2.addWeighted(resized, 1 - args.alpha, colorize_mask(mask), args.alpha, 0)
        report = runtime_report(selection, model.binding,
                                str(getattr(model.runner.runtime, 'version', 'unknown')),
                                args.test_img, mask, elapsed,
                                args.mask_save_path, args.img_save_path)
        save_results(args.mask_save_path, args.img_save_path, args.report_path,
                     mask, overlay, report)
        print(json.dumps(report, indent=2))
        return 0
    except (ValueError, OSError, RuntimeError, ImportError) as exc:
        print(f'error: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
