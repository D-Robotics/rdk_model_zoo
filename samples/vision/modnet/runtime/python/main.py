# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0

"""SDK-free CLI boundary and board execution entrypoint for MODNet."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

# Make direct ``python /abs/path/main.py`` work from any cwd without importing
# the board SDK or changing the host-safe list/dry-run paths.
_ROOT = Path(__file__).resolve().parents[5]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from samples.vision.modnet.runtime.python.model_binding import BindingError, SUPPORTED_TARGETS, list_available_assets, resolve_selection
from samples.vision.modnet.runtime.python.model_runner import RuntimeModelRunner
from samples.vision.modnet.runtime.python.modnet import MODNetTask
from samples.vision.modnet.runtime.python.visualization import composite


_SAMPLE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_TEST_IMAGE = _SAMPLE_DIR / "test_data" / "person.jpg"
DEFAULT_BG_IMAGE = _SAMPLE_DIR / "test_data" / "bg.jpg"
DEFAULT_MATTE_PATH = _SAMPLE_DIR / "test_data" / "matte.png"
DEFAULT_RESULT_PATH = _SAMPLE_DIR / "test_data" / "result.png"


def build_parser() -> argparse.ArgumentParser:
    """Build the host-safe MODNet CLI parser."""

    parser = argparse.ArgumentParser(description="MODNet portrait matting")
    parser.add_argument("--target", choices=("auto",) + SUPPORTED_TARGETS, default="auto")
    parser.add_argument("--asset-id", help="Exact manifest reference, for example x5:modnet:modnet_512x512_rgb.bin")
    parser.add_argument("--model-path", help="External model path; requires the exact --asset-id")
    parser.add_argument("--test-img", default=str(DEFAULT_TEST_IMAGE), help="BGR input image")
    parser.add_argument("--bg-img", default=str(DEFAULT_BG_IMAGE), help="Optional background for composite output")
    parser.add_argument("--matte-save-path", default=str(DEFAULT_MATTE_PATH))
    parser.add_argument("--img-save-path", default=str(DEFAULT_RESULT_PATH))
    parser.add_argument("--priority", type=int, default=0)
    parser.add_argument("--bpu-cores", nargs="+", type=int, default=[0])
    parser.add_argument("--ref-size", type=int, default=512, help="Compiled model input size; must remain 512")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--list-models", action="store_true")
    modes.add_argument("--dry-run", action="store_true")
    return parser


def _list_models(target: str) -> int:
    rows = list_available_assets(target)
    print(json.dumps([
        {"asset_id": row.reference, "filename": row.filename, "format": row.format,
         "url": row.url, "sha256": row.sha256, "target": "x5", "availability": "manual"}
        for row in rows
    ], ensure_ascii=False, indent=2))
    return 0


def _dry_run(args: argparse.Namespace) -> int:
    selection = resolve_selection(args.target, asset_id=args.asset_id, model_path=args.model_path)
    print(json.dumps({
        "target": selection.target, "asset_id": selection.asset.reference,
        "model_path": str(selection.model_path), "model_format": selection.asset.format,
        "input_shape": [1, 3, 512, 512], "input_dtype": "float32",
        "output_shape": [1, 1, 512, 512], "output_dtype": "float32",
        "input_protocol": "BGR HWC -> RGB F32 NCHW, normalize [-1,1], long-side resize + zero padding",
        "model_path_exists": selection.model_path.is_file(),
        "sdk_loaded": False, "downloaded": False,
    }, ensure_ascii=False, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    """Run list, dry-run, or board inference and return 0/2."""

    args = build_parser().parse_args(argv)
    try:
        if args.dry_run and args.ref_size != 512:
            raise ValueError("MODNet deployment metadata is fixed at ref-size 512.")
        if args.list_models:
            return _list_models(args.target)
        if args.dry_run:
            return _dry_run(args)
        if args.ref_size != 512:
            raise ValueError("MODNet deployment metadata is fixed at ref-size 512.")
        selection = resolve_selection(args.target, asset_id=args.asset_id, model_path=args.model_path)
        if not selection.model_path.is_file():
            raise FileNotFoundError(f"model file not found: {selection.model_path}")
        from samples._shared.platforms import require_execution_target
        require_execution_target(selection.target)
        import cv2
        image = cv2.imread(str(Path(args.test_img).expanduser()), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"input image not found or unreadable: {args.test_img}")
        runner = RuntimeModelRunner(selection)
        binding = runner.load()
        runner.set_scheduling_params(priority=args.priority, bpu_cores=args.bpu_cores)
        task = MODNetTask(runner, binding)
        prepared = task.pre_process(image)
        matte = task.post_process(task.forward(prepared.tensors), prepared.context)
        matte_path = Path(args.matte_save_path).expanduser()
        matte_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(matte_path), matte):
            raise OSError(f"failed to save matte: {matte_path}")
        result_path = None
        bg_path = Path(args.bg_img).expanduser()
        if bg_path.is_file():
            background = cv2.imread(str(bg_path), cv2.IMREAD_COLOR)
            if background is None:
                raise OSError(f"background image unreadable: {bg_path}")
            result = composite(image, matte, background)
            result_path = Path(args.img_save_path).expanduser()
            result_path.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(result_path), result):
                raise OSError(f"failed to save composite: {result_path}")
        print(json.dumps({"target": selection.target, "asset_id": selection.asset.reference,
                          "matte_path": str(matte_path), "composite_path": str(result_path) if result_path else None}, ensure_ascii=False))
        return 0
    except (BindingError, FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
