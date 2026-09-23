"""SDK-free CLI and board entrypoint for MobileSAM."""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[5]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from samples.vision.mobile_sam.runtime.python.model_binding import (  # noqa: E402
    SAMPLE_DIR,
    list_available_assets,
    resolve_selection,
)
from samples._shared.sam_tensor_io import validate_box  # noqa: E402

DEFAULT_TEST_IMAGE = SAMPLE_DIR / "test_data" / "dogs.jpg"
DEFAULT_RESULT_IMAGE = SAMPLE_DIR / "test_data" / "mobile_sam_full_mask_result.jpg"
DEFAULT_MASK_IMAGE = SAMPLE_DIR / "test_data" / "mobile_sam_binary_mask_result.png"
DEFAULT_BOX = (185.0, 120.0, 380.0, 445.0)


def parse_box(value: str) -> tuple[float, float, float, float]:
    try:
        values = validate_box(value.split(","))
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("box must be x1,y1,x2,y2") from exc
    return values  # type: ignore[return-value]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MobileSAM dual-model box-prompt mask segmentation.")
    parser.add_argument("--target", choices=("auto", "x5", "s100", "s100p", "s600"), default="auto")
    parser.add_argument("--encoder-asset-id", default=None, help="Exact manifest encoder asset reference.")
    parser.add_argument("--decoder-asset-id", default=None, help="Exact manifest decoder asset reference.")
    parser.add_argument("--encoder-model-path", default=None, help="External encoder model path; requires its asset ID.")
    parser.add_argument("--decoder-model-path", default=None, help="External decoder model path; requires its asset ID.")
    parser.add_argument("--test-img", default=str(DEFAULT_TEST_IMAGE), help="Input BGR image path.")
    parser.add_argument("--img-save-path", default=str(DEFAULT_RESULT_IMAGE), help="Overlay output path.")
    parser.add_argument("--mask-save-path", default=str(DEFAULT_MASK_IMAGE), help="Binary mask output path.")
    parser.add_argument("--box", type=parse_box, default=DEFAULT_BOX, help="Box x1,y1,x2,y2 in resized 512x512 coordinates.")
    parser.add_argument("--priority", type=int, default=0, help="Runtime scheduling priority.")
    parser.add_argument("--bpu-cores", nargs="+", type=int, default=None, help="S-series BPU core indexes; X5 rejects this option.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--list-models", action="store_true", help="List manifest encoder/decoder assets without loading SDK.")
    mode.add_argument("--dry-run", action="store_true", help="Resolve paths and tensor contract without loading SDK or files.")
    return parser


def _asset_ref(asset) -> str:
    return str(getattr(asset, "reference", getattr(asset, "asset_id", asset)))


def _selection_report(selection) -> dict:
    return {
        "target": selection.target,
        "encoder_asset_id": _asset_ref(selection.encoder_asset),
        "decoder_asset_id": _asset_ref(selection.decoder_asset),
        "encoder_model_path": str(selection.encoder_model_path),
        "decoder_model_path": str(selection.decoder_model_path),
    }


def _save_outputs(result: dict, image: np.ndarray, result_path: str, mask_path: str) -> None:
    visualization = importlib.import_module("samples.vision.mobile_sam.runtime.python.visualization")
    overlay = visualization.draw_mask_result(image, result["mask"], result["iou"], result["mask_index"])
    result_file = Path(result_path).expanduser()
    mask_file = Path(mask_path).expanduser()
    result_file.parent.mkdir(parents=True, exist_ok=True)
    mask_file.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(result_file), overlay) or not cv2.imwrite(str(mask_file), result["mask"].astype(np.uint8) * 255):
        raise OSError("failed to write MobileSAM output image")


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.list_models:
            assets = list_available_assets(args.target)
            for asset in assets:
                print(_asset_ref(asset))
            print(f"{len(assets)} manifest assets; no model loaded.")
            return 0
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
            report = _selection_report(selection)
            report["input"] = {"shape": [1, 3, 512, 512], "dtype": "float32", "layout": "RGB NCHW"}
            report["decoder"] = {
                "embedding": [1, 256, 32, 32],
                "boxes_shapes": [[1, 4], [1, 4, 1, 1]] if selection.target == "x5" else [[1, 4]],
                "box_shape_status": "requires runtime metadata",
                "box": args.box, "mask_candidates": 3, "mask_semantics": "raw logits",
            }
            print(json.dumps(report, indent=2))
            return 0
        image = cv2.imread(str(Path(args.test_img).expanduser()), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(args.test_img)
        runner = importlib.import_module("samples.vision.mobile_sam.runtime.python.model_runner").RuntimeModelRunner(selection)
        binding = runner.load()
        cores = args.bpu_cores if args.bpu_cores is not None else ([0] if selection.target in ("s100", "s100p", "s600") else None)
        runner.set_scheduling_params(priority=args.priority, bpu_cores=cores)
        pipeline = importlib.import_module("samples.vision.mobile_sam.runtime.python.pipeline").MobileSAMPipeline(runner, binding)
        result = pipeline.predict(image, box=args.box)
        _save_outputs(result, image, args.img_save_path, args.mask_save_path)
        print(json.dumps({**_selection_report(selection), "image": str(Path(args.test_img).expanduser()), "mask_path": str(Path(args.mask_save_path).expanduser()), "iou": float(result["iou"]), "mask_index": int(result["mask_index"])}, indent=2))
        return 0
    except (ImportError, OSError, RuntimeError, ValueError, TypeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
