# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""CLIP native CLI: prepare an exact encoder pair and match text to one image.

This file stays deliberately small: parse the arguments, handle the
model-free listing/dry-run modes, resolve the pair, construct the task,
call ``predict``, show the result.  Option declarations, prompt parsing,
result presentation and the annotated-image write live in ``cli.py``; the
matching flow itself (preprocess → infer → postprocess) lives in
``matching.py``.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from samples.vision.clip.runtime.python.cli import (  # noqa: E402
    build_parser,  # re-exported here: the contract checker imports it from main
    parse_prompts,
    print_match_result,
    read_bgr_image,
    run_dry_run,
    run_list_models,
    save_annotated_image,
)
from samples.vision.clip.runtime.python.model_binding import (  # noqa: E402
    list_available_assets,  # noqa: F401 - import path kept for existing callers
    resolve_selection,
)


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        if args.list_models:
            return run_list_models(args.target)
        if args.dry_run and args.target == 'auto':
            raise ValueError('Host dry-run requires --target x5; no board detection performed.')
        selection = resolve_selection(args.target, image_asset_id=args.image_asset_id,
                                      text_asset_id=args.text_asset_id,
                                      image_model_path=args.image_model_path,
                                      text_model_path=args.text_model_path)
        if args.dry_run:
            return run_dry_run(selection)
        from samples._shared.platforms import require_execution_target
        require_execution_target(selection.target)
        for path in (selection.image_model_path, selection.text_model_path):
            if not path.is_file():
                raise FileNotFoundError(f'Model not found: {path}; prepare the pair with model/download.sh.')

        # Imported inside real execution: OpenCV, onnxruntime and hbm_runtime
        # load only after the selection, file, and board checks above passed.
        from samples.vision.clip.runtime.python.matching import CLIPTask
        from samples.vision.clip.runtime.python.model_runner import RuntimeModelRunner
        from samples.vision.clip.runtime.python.tokenization import PromptTokenizer

        prompts = parse_prompts(args.texts)
        image = read_bgr_image(args.test_img)
        runner = RuntimeModelRunner(selection)
        binding = runner.load()
        runner.set_scheduling_params(priority=args.priority, bpu_cores=args.bpu_cores)
        task = CLIPTask(runner, binding, PromptTokenizer())
        result = task.predict(image, prompts)
        save_annotated_image(args.img_save_path, image, prompts, result)
        print_match_result(result, selection, prompts, image_saved=args.img_save_path)
        return 0
    except (ImportError, OSError, ValueError, RuntimeError) as exc:
        print(f'error: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
