# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Native SigLIP CLI: model selection, image/file I/O and feature summaries.

This file stays deliberately small: parse the arguments, handle the
model-free listing/dry-run modes, resolve the selection, construct the
task, call ``predict``, show the result.  Option declarations, the
model-free modes, the feature summary and the optional NumPy save live in
``cli.py``; the feature flow itself (preprocess → infer → postprocess)
lives in ``embedding.py``.
"""
from __future__ import annotations
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[5]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from samples.vision.siglip.runtime.python.cli import (  # noqa: E402
    build_parser,  # re-exported here: existing callers import it from main
    print_summary,
    read_bgr_image,
    run_dry_run,
    run_list_models,
    save_feature_tensor,
    summarize_result,
)
from samples.vision.siglip.runtime.python.model_binding import (  # noqa: E402
    list_available_assets,  # noqa: F401 - import path kept for existing callers
    resolve_selection,
)


def main(argv=None) -> int:
    """Print resolved contracts or execute one feature extraction; return 0/2."""
    args = build_parser().parse_args(argv)
    try:
        if args.list_models:
            return run_list_models(args.target)
        if args.dry_run and args.target == 'auto':
            raise ValueError('Host dry-run requires --target s100 or --target s100p; no board detection performed.')
        selection = resolve_selection(args.target, variant=args.variant, asset_id=args.asset_id,
            model_path=args.model_path, submodel=args.submodel, image_size=args.image_size)
        if args.dry_run:
            return run_dry_run(selection)
        from samples._shared.platforms import require_execution_target
        require_execution_target(selection.target)
        if not selection.model_path.is_file():
            raise FileNotFoundError(f'Model not found: {selection.model_path}; prepare it explicitly with model/download.sh.')

        # Imported inside real execution: OpenCV and hbm_runtime load only
        # after the selection, file, and board checks above have passed.
        from samples.vision.siglip.runtime.python.embedding import SigLIPTask
        from samples.vision.siglip.runtime.python.model_runner import RuntimeModelRunner

        image = read_bgr_image(args.test_img)
        runner = RuntimeModelRunner(selection)
        binding = runner.load()
        runner.set_scheduling_params(priority=args.priority, bpu_cores=args.bpu_cores)
        result = SigLIPTask(runner, binding).predict(image)
        print_summary(summarize_result(result, selection.submodel))
        if args.output_file is not None:
            save_feature_tensor(args.output_file, result)
        return 0
    except (OSError, ValueError, RuntimeError) as exc:
        print(f'error: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
