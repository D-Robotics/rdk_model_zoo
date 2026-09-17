# Copyright (c) 2026 D-Robotics Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Native entrypoint for the bounded Python PaddleOCR composition pilot.

The entrypoint is the only module that adjusts ``sys.path`` for direct
full-checkout invocation.  SDK, OpenCV and pyclipper imports stay behind the
selected operation so help/list/dry-run remain host-safe.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Optional, Sequence


_ROOT = Path(__file__).resolve().parents[5]
if str(_ROOT) not in sys.path:
    # ``python samples/vision/paddle_ocr/runtime/python/main.py`` is supported
    # from any current directory inside a full source checkout.
    sys.path.insert(0, str(_ROOT))

from samples.vision.paddle_ocr.runtime.python.model_binding import (  # noqa: E402
    BindingError,
    OCRPair,
    SUPPORTED_TARGETS,
    list_available_pairs,
    resolve_pair,
)


_SAMPLE_DIR = _ROOT / "samples" / "vision" / "paddle_ocr"


def build_parser() -> argparse.ArgumentParser:
    """Build the SDK-free parser; ``main`` returns zero or a user error code."""

    parser = argparse.ArgumentParser(
        description="Audited two-stage PaddleOCR inference on RDK X5 or S100."
    )
    parser.add_argument(
        "--target",
        choices=("auto",) + SUPPORTED_TARGETS,
        default="auto",
        help="Target selection: auto, x5, s100, s100p, or s600.",
    )
    parser.add_argument(
        "--det-asset-id",
        help="Qualified detector manifest reference group:sample:filename.",
    )
    parser.add_argument(
        "--rec-asset-id",
        help="Qualified recognizer manifest reference group:sample:filename.",
    )
    parser.add_argument(
        "--det-model-path",
        help="Existing local detector artifact; never downloaded implicitly.",
    )
    parser.add_argument(
        "--rec-model-path",
        help="Existing local recognizer artifact; never downloaded implicitly.",
    )
    parser.add_argument(
        "--vocabulary-path",
        help="Optional checked-in S100 UTF-8 dictionary replacement (hash checked).",
    )
    parser.add_argument(
        "--test-img",
        help="BGR input image path (defaults to the selected sample fixture).",
    )
    parser.add_argument(
        "--output-format",
        choices=("text", "json"),
        default="text",
        help="Inference/report output format (default: text).",
    )
    parser.add_argument(
        "--json-output",
        help="Also write the JSON inference result to this local path.",
    )
    parser.add_argument(
        "--priority",
        type=int,
        default=0,
        help="Runtime scheduling priority, 0 through 255 (default: 0).",
    )
    parser.add_argument(
        "--bpu-cores",
        nargs="+",
        type=int,
        default=[0],
        help="Runtime BPU core indexes (default: 0).",
    )
    parser.add_argument(
        "--model-dir",
        help="Destination directory for the explicit --prepare operation.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--list-models",
        action="store_true",
        help="List finite manifest-backed detector/recognizer pairs.",
    )
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve a pair and print its static contract without SDK access.",
    )
    mode.add_argument(
        "--prepare",
        action="store_true",
        help="Explicitly fetch selected manifest assets into local paths.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Execute list, dry-run, prepare or inference and return a process code."""

    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.list_models:
            return _list_models(args.target, args.output_format)
        if args.dry_run:
            return _dry_run(args)
        if args.prepare:
            return _prepare(args)

        pair = _resolve_pair_from_args(args, for_execution=True)
        detector_path = pair.detector_model_path
        recognizer_path = pair.recognizer_model_path
        if not detector_path.is_file():
            raise FileNotFoundError(f"detector model file not found: {detector_path}")
        if not recognizer_path.is_file():
            raise FileNotFoundError(f"recognizer model file not found: {recognizer_path}")

        # A real execution must prove the exact detected board before importing
        # hbm_runtime.  The runner repeats this check immediately before load.
        from samples._shared.platforms import require_execution_target

        require_execution_target(pair.target)
        return _run(pair, args)
    except (BindingError, FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


def _list_models(target: str, output_format: str) -> int:
    pairs = list_available_pairs(target)
    records = [_pair_record(pair) for pair in pairs]
    if output_format == "json":
        print(json.dumps(records, ensure_ascii=False, indent=2))
        return 0
    print("Manifest-backed PaddleOCR pilot pair references:")
    if not records:
        print(f"  no audited pair for target: {target}")
    for record in records:
        print(f"  {record['target']} / {record['variant']}")
        print(f"    detector:   {record['det_asset_id']}")
        print(f"    recognizer: {record['rec_asset_id']}")
    print("References qualify existing manifest rows; they are not standalone catalog IDs.")
    return 0


def _dry_run(args: argparse.Namespace) -> int:
    if args.target == "auto" and args.det_asset_id is None and args.rec_asset_id is None:
        candidates = [_pair_record(pair) for pair in list_available_pairs("auto")]
        if args.output_format == "json":
            print(json.dumps({"candidates": candidates}, ensure_ascii=False, indent=2))
        else:
            print("Dry-run needs --target x5/s100 or both qualified asset references.")
            for record in candidates:
                print(f"  {record['target']}: {record['det_asset_id']} + {record['rec_asset_id']}")
            print("No model is downloaded and no SDK/pyclipper is loaded.")
        return 0
    pair = _resolve_pair_from_args(args)
    record = _pair_record(pair)
    if args.output_format == "json":
        print(json.dumps(record, ensure_ascii=False, indent=2))
    else:
        print("Dry-run selection:")
        for key, value in record.items():
            if isinstance(value, (dict, list)):
                value = json.dumps(value, ensure_ascii=False, sort_keys=True)
            print(f"  {key}: {value}")
        print("No model is downloaded and no SDK/pyclipper is loaded.")
    return 0


def _prepare(args: argparse.Namespace) -> int:
    """Perform the only network-capable operation, when explicitly requested."""

    if args.det_asset_id is None or args.rec_asset_id is None:
        raise BindingError("--prepare requires both --det-asset-id and --rec-asset-id.")
    pair = _resolve_pair_from_args(args)
    from samples._shared.assets import download_asset, resolve_asset

    destinations = [
        Path(args.det_model_path).expanduser()
        if args.det_model_path
        else pair.detector_model_path,
        Path(args.rec_model_path).expanduser()
        if args.rec_model_path
        else pair.recognizer_model_path,
    ]
    references = [pair.detector_asset, pair.recognizer_asset]
    if args.model_dir:
        root = Path(args.model_dir).expanduser()
        destinations = [root / Path(destination).name for destination in destinations]
    if destinations[0].resolve() == destinations[1].resolve():
        raise BindingError(
            "--prepare needs distinct detector and recognizer destination files."
        )
    observed = []
    for reference, destination in zip(references, destinations):
        observed.append(
            {
                "asset_id": reference,
                "path": str(destination),
                "sha256": download_asset(resolve_asset(reference), destination),
            }
        )
    payload = {"prepared": observed, "target": pair.target}
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def _run(pair: OCRPair, args: argparse.Namespace) -> int:
    import cv2

    from samples.vision.paddle_ocr.runtime.python.model_runner import create_stage_runners
    from samples.vision.paddle_ocr.runtime.python.pipeline import OCRPipeline

    image_path = (
        Path(args.test_img).expanduser()
        if args.test_img
        else _default_image(pair.target)
    )
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"could not read BGR image: {image_path}")
    detector, recognizer = create_stage_runners(
        pair,
        priority=args.priority,
        bpu_cores=args.bpu_cores,
    )
    result = OCRPipeline(
        pair,
        detector,
        recognizer,
        vocabulary_path=args.vocabulary_path,
    ).predict(image)
    payload = result.as_dict()
    payload["image_shape"] = list(image.shape)
    if args.output_format == "json":
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(f"target: {payload['target']}")
        for index, (box, text) in enumerate(zip(payload["boxes"], payload["texts"])):
            print(f"[{index}] text={text!r} box={box}")
    if args.json_output:
        output_path = Path(args.json_output).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return 0


def _resolve_pair_from_args(
    args: argparse.Namespace,
    *,
    for_execution: bool = False,
) -> OCRPair:
    target = args.target
    if (
        for_execution
        and target == "auto"
        and args.det_asset_id is None
        and args.rec_asset_id is None
    ):
        # The library resolver stays host-independent and requires explicit
        # qualified refs for auto.  The native execution entrypoint may first
        # identify this board, then select that target's default audited pair.
        from samples._shared.platforms import resolve_target

        target = resolve_target("auto")
    return resolve_pair(
        target,
        det_asset_id=args.det_asset_id,
        rec_asset_id=args.rec_asset_id,
        det_model_path=args.det_model_path,
        rec_model_path=args.rec_model_path,
    )


def _default_image(target: str) -> Path:
    if target == "x5":
        return _SAMPLE_DIR / "test_data" / "x5" / "paddleocr_test.jpg"
    return _SAMPLE_DIR / "test_data" / "s100" / "gt_2322.jpg"


def _pair_record(pair: OCRPair) -> dict[str, Any]:
    return {
        "target": pair.target,
        "variant": pair.variant,
        "det_asset_id": pair.detector_asset,
        "rec_asset_id": pair.recognizer_asset,
        "det_model_path": str(pair.detector_model_path),
        "rec_model_path": str(pair.recognizer_model_path),
        "detector": _contract_record(pair.detector),
        "recognizer": _contract_record(pair.recognizer),
        "vocabulary": {
            "kind": pair.vocabulary.kind,
            "class_count": pair.vocabulary.class_count,
            "sha256": pair.vocabulary.sha256,
        },
    }


def _contract_record(contract: Any) -> dict[str, Any]:
    return {
        "model_name": contract.model_name,
        "input_names": list(contract.input_names),
        "input_shapes": {
            key: list(value) for key, value in contract.input_shapes.items()
        },
        "input_dtypes": dict(contract.input_dtypes),
        "runtime_input_shapes": {
            key: list(value) for key, value in contract.runtime_input_shapes.items()
        },
        "runtime_input_dtypes": dict(contract.runtime_input_dtypes),
        "output_name": contract.output_name,
        "output_shape": list(contract.output_shape),
        "output_dtype": contract.output_dtype,
        "input_protocol": contract.input_protocol,
        "output_semantics": contract.output_semantics,
    }


if __name__ == "__main__":
    raise SystemExit(main())
