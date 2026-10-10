"""Run the YOLOv5 command-line sample.

Parse options, construct the runner and the detection task, call ``predict``,
and present the result. Option declarations, listing/dry-run, and rendering
live in ``cli.py``; the detection stages and per-call geometry live in
``detection.py``; published manifest selection in ``cli.py`` and tensor
contracts in ``detection.py``. Help, listing, and dry-run stay model-free.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from samples.vision.yolov5.runtime.python.cli import (  # noqa: F401 - build_parser re-exported for the contract checker
    build_parser,
    default_image_path,
    draw_detections,
    dry_run,
    list_models,
    validate_options,
)
from samples.vision.yolov5.runtime.python.cli import resolve_selection


def main(argv=None) -> int:
    """Run the requested YOLOv5 command and return its exit status.

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
        validate_options(args)
        selection = resolve_selection(args.target, variant=args.variant, asset_id=args.asset_id, model_path=args.model_path)
        image_path = default_image_path(args, selection)
        if args.dry_run:
            return dry_run(selection, image_path)
        import cv2
        from samples.vision.yolov5.runtime.python.detection import YOLOv5Task
        task = YOLOv5Task(selection, score_thres=args.score_thres, nms_thres=args.nms_thres, anchors=args.anchors)
        task.set_scheduling_params(priority=args.priority, bpu_cores=args.bpu_cores)
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f'Cannot read image: {image_path}')
        labels = Path(args.label_file).expanduser().read_text().splitlines()
        result = task.predict(image, resize_type=args.resize_type)
        path = Path(args.img_save_path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(path), draw_detections(image, result, labels)):
            raise OSError(f'Cannot save: {path}')
        print(json.dumps(dict(boxes=result.boxes.tolist(), scores=result.scores.tolist(), class_ids=result.class_ids.tolist()), ensure_ascii=False))
        print(f'Saved {len(result.scores)} detections to {path}')
        return 0
    except (ValueError, OSError, RuntimeError, ImportError) as exc:
        print(f'Error: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
