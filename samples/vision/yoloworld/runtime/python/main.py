"""Run the YOLOWorld command-line sample.

Parse options, construct the runner and the open-vocabulary task, call
``predict``, and present the result. Option declarations, published asset
selection, listing/dry-run, prompt parsing, and drawing live in ``cli.py``;
the task stages and tensor binding live in ``yoloworld.py``.
Help, listing, and dry-run stay model-free.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from samples.vision.yoloworld.runtime.python.cli import (  # noqa: F401 - build_parser re-exported for the contract checker
    build_parser,
    draw_results,
    dry_run,
    list_models,
    parse_prompts,
    save_image,
)
from samples.vision.yoloworld.runtime.python.cli import resolve_selection


def main(argv=None) -> int:
    """Run the requested YOLOWorld command and return its exit status.

    Args:
        argv: Optional argument sequence excluding the program name; None
            reads sys.argv through argparse.

    Returns:
        int: 0 for a successful run; 2 for a reported selection, option,
        IO, or runtime error.

    Raises:
        SystemExit: argparse handles --help or rejects invalid arguments.
    """
    args = build_parser().parse_args(argv)
    try:
        if args.list_models:
            return list_models(args.target)
        if args.dry_run and args.target == 'auto':
            raise ValueError('Host dry-run requires explicit --target x5; it never probes a board.')
        prompts = parse_prompts(args.prompts)
        selection = resolve_selection(args.target, model_path=args.model_path, asset_id=args.asset_id)
        if args.dry_run:
            return dry_run(selection, prompts, args)
        from utils.py_utils.platforms import require_execution_target
        require_execution_target(selection.target)
        if not selection.model_path.is_file():
            raise FileNotFoundError(f'Model not found: {selection.model_path}; run model/download.sh explicitly.')
        import cv2
        from samples.vision.yoloworld.runtime.python.yoloworld import YOLOWorldTask
        image = cv2.imread(str(Path(args.test_img).expanduser()), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f'Cannot read image: {args.test_img}')
        with Path(args.vocab_file).expanduser().open(encoding='utf-8') as handle:
            vocabulary = json.load(handle)
        task = YOLOWorldTask(selection, vocabulary, score_thres=args.score_thres, nms_thres=args.nms_thres)
        task.set_scheduling_params(priority=args.priority, bpu_cores=args.bpu_cores)
        result = task.predict(image, prompts)
        save_image(args.img_save_path, draw_results(image, result, task.class_names))
        print(json.dumps({'target': selection.target, 'prompts': prompts, 'count': int(len(result.scores)),
                          'image_saved': str(Path(args.img_save_path).expanduser())}, indent=2))
        return 0
    except (OSError, KeyError, TypeError, ValueError, RuntimeError, ImportError) as exc:
        print(f'error: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
