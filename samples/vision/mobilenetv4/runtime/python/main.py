"""People and Agent entrypoint for the MobileNetV4 classification sample.

This file stays deliberately small: parse the arguments, resolve the model,
construct the classifier, call ``predict``, show the result.  Option
declarations and the model-free listing/dry-run modes live in ``cli.py``;
the classification flow itself (preprocess → infer → postprocess) lives in
``classify.py``.  Listing and dry-run are deliberately model-free; model
preparation remains an explicit user action.
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Sequence

_ROOT = Path(__file__).resolve().parents[5]
if str(_ROOT) not in sys.path:
    # Direct ``python samples/.../main.py`` is a supported full-checkout entry.
    # This compatibility adjustment is kept at the entry boundary only.
    sys.path.insert(0, str(_ROOT))

from samples.vision.mobilenetv4.runtime.python.cli import (  # noqa: E402
    build_parser,  # re-exported here: the contract checker imports it from main
    default_labels,
    print_classification_result,
    read_bgr_image,
    run_dry_run,
    run_list_models,
    save_result_image,
)
from samples.vision.mobilenetv4.runtime.python.model_binding import (  # noqa: E402
    BindingError,
    resolve_selection,
)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the requested MobileNetV4 command and return its exit status.

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
    if args.list_models:
        return run_list_models(args.target)
    if args.dry_run:
        return run_dry_run(args)

    try:
        selection = resolve_selection(
            args.target,
            asset_id=args.asset_id,
            variant=getattr(args, "variant", None),
            model_path=args.model_path,
        )
        if not selection.model_path.is_file():
            raise FileNotFoundError(f"model file not found: {selection.model_path}")

        # A path existing on disk is not execution authorization.  Require the
        # exact detected board before touching hbm_runtime.
        from utils.py_utils.platforms import require_execution_target

        require_execution_target(selection.target)

        # Imported inside real execution: OpenCV and hbm_runtime load only
        # after the selection, file, and board checks above have passed.
        from samples.vision.mobilenetv4.runtime.python.classify import MobileNetV4Classifier

        model = MobileNetV4Classifier(
            selection,
            top_k=args.top_k,
            labels=default_labels(args.label_file),
            resize_type=args.resize_type,
        )
        model.set_scheduling_params(priority=args.priority, bpu_cores=args.bpu_cores)

        result = model.predict(args.test_img)
        print_classification_result(result, selection, top_k=args.top_k)

        if args.img_save_path:
            save_result_image(
                Path(args.img_save_path).expanduser(),
                read_bgr_image(args.test_img),
                result,
            )
            print(f"Saved result image: {args.img_save_path}")
        return 0
    except (BindingError, FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
