"""SDK-free CLI and board entrypoint for MobileSAM.

This file stays deliberately small: parse the arguments, handle the
model-free listing/dry-run modes, resolve the selection, construct the
encoder/decoder runners and the pipeline, call ``predict`` with the box
prompt, show the result.  Option declarations (including the box parser),
the model-free modes, image reading and the overlay/mask writes live in
``cli.py``; the encoder → decoder composition lives in ``pipeline.py``.
"""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[5]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from samples.vision.mobile_sam.runtime.python.cli import (  # noqa: E402
    build_parser,  # re-exported here: existing callers import it from main
    dry_run_report,
    parse_box,  # re-exported here: the established entry surface keeps it
    print_report,
    read_bgr_image,
    run_list_models,
    save_outputs,
    selection_report,
)
from samples.vision.mobile_sam.runtime.python.model_binding import (  # noqa: E402
    list_available_assets,  # noqa: F401 - import path kept for existing callers
    resolve_selection,
)


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.list_models:
            return run_list_models(args.target)
        if args.dry_run and args.target == "auto":
            raise ValueError("--dry-run requires an explicit --target; auto target detection needs a board identity.")
        selection = resolve_selection(
            args.target,
            encoder_model_path=args.encoder_model_path,
            decoder_model_path=args.decoder_model_path,
            encoder_asset_id=args.encoder_asset_id,
            decoder_asset_id=args.decoder_asset_id,
        )
        if args.bpu_cores is not None and selection.target == "x5":
            raise ValueError("--bpu-cores is not supported on X5.")
        if args.dry_run:
            print_report(dry_run_report(selection, args.box))
            return 0

        # A real execution must prove the exact detected board before any SDK
        # import; the stage runners repeat this check immediately before load.
        from utils.py_utils.platforms import require_execution_target

        require_execution_target(selection.target)

        # Imported inside real execution: OpenCV and hbm_runtime load only
        # after the selection and board checks above have passed.
        from utils.py_utils.sam_runner import RuntimeModelRunner
        from samples.vision.mobile_sam.runtime.python.pipeline import MobileSAMPipeline

        image = read_bgr_image(args.test_img)
        runner = RuntimeModelRunner(selection)
        binding = runner.load()
        cores = args.bpu_cores if args.bpu_cores is not None else (
            [0] if selection.target in ("s100", "s100p", "s600") else None)
        runner.set_scheduling_params(priority=args.priority, bpu_cores=cores)
        pipeline = MobileSAMPipeline(runner, binding)
        result = pipeline.predict(image, box=args.box)
        save_outputs(result, image, args.img_save_path, args.mask_save_path)
        print_report({**selection_report(selection), "image": str(Path(args.test_img).expanduser()),
                      "mask_path": str(Path(args.mask_save_path).expanduser()),
                      "iou": float(result["iou"]), "mask_index": int(result["mask_index"])})
        return 0
    except (ImportError, OSError, RuntimeError, ValueError, TypeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
