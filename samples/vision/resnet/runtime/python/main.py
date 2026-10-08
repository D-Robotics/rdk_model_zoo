"""Run the ResNet command-line sample.

Parse options, construct ResNetClassifier, call predict, and present results.
CLI helpers supply model listing, dry-run, label loading, and image output.
"""


from __future__ import annotations

from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[5]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from utils.py_utils.cls_binding import BindingError
from utils.py_utils.image import read_bgr_image
from utils.py_utils.labels import load_labels as _load_labels
from samples.vision.resnet.runtime.python.classify import ResNetClassifier
from samples.vision.resnet.runtime.python.cli import (
    build_parser, list_models, dry_run, resolve_selection, print_result, save_result_image,
)

_DEFAULT_LABELS = _ROOT / "datasets/imagenet/imagenet_classes.names"


def main(argv=None) -> int:
    """Run the requested ResNet command and return its exit status.

    Args:
        argv: Optional command-line argument sequence, excluding the program name.
            None reads sys.argv through argparse.

    Returns:
        int: 0 for success; 2 for a reported selection, IO, or runtime error.

    Raises:
        SystemExit: argparse handles --help or rejects invalid arguments.

    Notes:
        List and dry-run modes do not load a model. Inference prints Top-K
        results and writes an annotated image only when --img-save-path is set.
    """
    args = build_parser().parse_args(argv)
    try:
        if args.list_models:
            return list_models(args.target)
        if args.dry_run:
            return dry_run(args)
        label_path = Path(args.label_file).expanduser() if args.label_file else _DEFAULT_LABELS
        labels = _load_labels(label_path) if args.label_file or label_path.is_file() else None
        selection = resolve_selection(args.target, variant=args.variant,
                                      asset_id=args.asset_id, model_path=args.model_path)
        contract = selection.contract
        model = ResNetClassifier(
            selection.model_path, target=selection.target, top_k=args.top_k, labels=labels,
            resize_type=contract.resize_type if args.resize_type is None else args.resize_type,
            resize_interpolation=contract.resize_interpolation,
            score_policy=contract.output_score_policy)
        model.set_scheduling_params(priority=args.priority, bpu_cores=args.bpu_cores)
        result = model.predict(args.test_img)
        print_result(result, selection, args.top_k)
        if args.img_save_path:
            save_result_image(Path(args.img_save_path).expanduser(), read_bgr_image(args.test_img), result)
            print(f"Saved result image: {args.img_save_path}")
        return 0
    except (BindingError, OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2



if __name__ == "__main__":
    raise SystemExit(main())
