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

"""Command-line surface for the PaddleOCR sample.

Option declarations, the model-free listing/dry-run modes, the explicit
``--prepare`` fetch operation and result rendering live here so
``main.py`` can stay a thin, readable entry: parse arguments, resolve the
pair, construct the two-stage pipeline, call ``predict``, show the result.
Nothing in this module runs OCR or loads a board SDK.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Optional, Sequence

from samples.vision.paddle_ocr.runtime.python.model_binding import (
    BindingError,
    OCRPair,
    SUPPORTED_TARGETS,
    list_available_pairs,
    resolve_pair,
)


_ROOT = Path(__file__).resolve().parents[5]
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


def resolve_pair_from_args(
    args: argparse.Namespace,
    *,
    for_execution: bool = False,
) -> OCRPair:
    """Resolve the pair for one command line; execution may detect the board."""

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


def run_list_models(target: str, output_format: str) -> int:
    """Print the manifest-backed pairs for ``target`` (model-free)."""

    pairs = list_available_pairs(target)
    records = [pair_record(pair) for pair in pairs]
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


def run_dry_run(args: argparse.Namespace) -> int:
    """Resolve and print one static pair contract without models or SDK."""

    if args.target == "auto" and args.det_asset_id is None and args.rec_asset_id is None:
        candidates = [pair_record(pair) for pair in list_available_pairs("auto")]
        if args.output_format == "json":
            print(json.dumps({"candidates": candidates}, ensure_ascii=False, indent=2))
        else:
            print("Dry-run needs --target x5/s100 or both qualified asset references.")
            for record in candidates:
                print(f"  {record['target']}: {record['det_asset_id']} + {record['rec_asset_id']}")
            print("No model is downloaded and no SDK/pyclipper is loaded.")
        return 0
    pair = resolve_pair_from_args(args)
    record = pair_record(pair)
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


def run_prepare(args: argparse.Namespace) -> int:
    """Perform the only network-capable operation, when explicitly requested."""

    if args.det_asset_id is None or args.rec_asset_id is None:
        raise BindingError("--prepare requires both --det-asset-id and --rec-asset-id.")
    pair = resolve_pair_from_args(args)
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


def default_image(target: str) -> Path:
    """Pick the bundled fixture of the resolved target."""

    if target == "x5":
        return _SAMPLE_DIR / "test_data" / "x5" / "paddleocr_test.jpg"
    return _SAMPLE_DIR / "test_data" / "s100" / "gt_2322.jpg"


def read_bgr_image(path: "str | Path"):
    """Read one BGR image; failures name the exact path."""

    import cv2

    resolved = Path(path).expanduser()
    image = cv2.imread(str(resolved), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"could not read BGR image: {resolved}")
    return image


def print_result(payload: dict[str, Any], output_format: str) -> None:
    """Render one finished prediction as text or JSON (presentation only)."""

    if output_format == "json":
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return
    print(f"target: {payload['target']}")
    for index, (box, text) in enumerate(zip(payload["boxes"], payload["texts"])):
        print(f"[{index}] text={text!r} box={box}")


def write_json_output(path: "str | Path", payload: dict[str, Any]) -> None:
    """Write the JSON result to exactly the requested local path."""

    output_path = Path(path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def pair_record(pair: OCRPair) -> dict[str, Any]:
    """Serializable description of one resolved pair and its contracts."""

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


__all__ = [
    "build_parser",
    "default_image",
    "pair_record",
    "print_result",
    "read_bgr_image",
    "resolve_pair_from_args",
    "run_dry_run",
    "run_list_models",
    "run_prepare",
    "write_json_output",
]
