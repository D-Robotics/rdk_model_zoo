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
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Literal, Mapping, Optional, Sequence

from utils.py_utils.assets import resolve_asset
from utils.py_utils.runtime_meta import (
    MetadataMismatchError as _SharedMetadataMismatchError,
)
from utils.py_utils.runtime_meta import RuntimeMetadata, canonicalise_dtype

# ======================================================================
# Published pair identity and listing.
# ======================================================================

SUPPORTED_TARGETS = ("x5", "s100", "s100p", "s600")
SUPPORTED_VARIANTS = ("ppocrv3", "ppocrv6")
_SUPPORTED_PUBLISHED_TARGETS = ("x5", "s100")
_SAMPLE_DIR = Path(__file__).resolve().parents[2]
_MODEL_DIR = _SAMPLE_DIR / "model"
_TEST_DATA_DIR = _SAMPLE_DIR / "test_data"
S100_VOCABULARY_SHA256 = (
    "b5f2bfe2bdd9448429e3e82b51c789775d9b42f2403d082b00662eb77e401c5d"
)


@dataclass(frozen=True)
class StageContract:
    """Static facts observed for one detector or recognizer artifact."""

    stage: Literal["detector", "recognizer"]
    asset_id: str
    target: str
    variant: str
    model_name: str
    input_names: tuple[str, ...]
    input_shapes: Mapping[str, tuple[int, ...]]
    input_dtypes: Mapping[str, str]
    runtime_input_shapes: Mapping[str, tuple[int, ...]]
    runtime_input_dtypes: Mapping[str, str]
    output_name: str
    output_shape: tuple[int, ...]
    output_dtype: str
    input_protocol: str
    output_semantics: str = "unverified_score_vector"

X5_ALPHABET = (
    "0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz"
    "{|}~!\"#$%&'()*+,-./  "
)


class BindingError(ValueError):
    """Base class for selection, metadata and tensor contract failures."""


class UnsupportedAssetError(BindingError):
    """The requested asset pair or target is outside the pilot boundary."""


class MetadataMismatchError(BindingError, _SharedMetadataMismatchError):
    """A loaded model or runner tensor does not satisfy its bound contract."""


@dataclass(frozen=True)
class VocabularySpec:
    """Vocabulary identity and construction policy for a recognizer pair."""

    kind: Literal["fixed", "utf8_lines"]
    class_count: int
    source_path: Optional[Path] = None
    sha256: Optional[str] = None

    def load_tokens(self, path: Optional[str | Path] = None) -> tuple[str, ...]:
        """Load and validate the token table used by the bound recognizer."""

        if self.kind == "fixed":
            if self.class_count != len(X5_ALPHABET) + 1:
                raise BindingError("The fixed X5 vocabulary contract is inconsistent.")
            return ("blank",) + tuple(X5_ALPHABET)

        actual_path = Path(path).expanduser() if path is not None else self.source_path
        if actual_path is None:
            raise BindingError("The S100 vocabulary path is required.")
        if not actual_path.is_file():
            raise FileNotFoundError(f"vocabulary file not found: {actual_path}")
        raw = actual_path.read_bytes()
        observed = hashlib.sha256(raw).hexdigest()
        if self.sha256 is not None and observed != self.sha256.lower():
            raise BindingError(
                f"vocabulary SHA-256 mismatch for {actual_path}: {observed}"
            )
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise BindingError(f"vocabulary is not valid UTF-8: {actual_path}") from exc
        # ``splitlines`` removes line terminators while keeping each source line
        # as one token.  The source entry prepends blank and appends one space.
        lines = tuple(text.splitlines())
        tokens = ("blank",) + lines + (" ",)
        if len(tokens) != self.class_count:
            raise BindingError(
                f"vocabulary class count {len(tokens)} does not match the bound "
                f"contract {self.class_count} for {actual_path}"
            )
        return tokens


@dataclass(frozen=True)
class OCRPair:
    """One detector/recognizer pair selected from existing manifest rows."""

    target: str
    variant: str
    detector_asset: str
    recognizer_asset: str
    detector: StageContract
    recognizer: StageContract
    detector_model_path: Path
    recognizer_model_path: Path
    vocabulary: VocabularySpec

    @property
    def det_model_path(self) -> Path:
        """Compatibility spelling for callers that use detector shorthand."""

        return self.detector_model_path

    @property
    def rec_model_path(self) -> Path:
        """Compatibility spelling for callers that use recognizer shorthand."""

        return self.recognizer_model_path

    @property
    def asset_ids(self) -> tuple[str, str]:
        """Return the ordered detector and recognizer qualified references."""

        return self.detector_asset, self.recognizer_asset


def list_available_pairs(target: Optional[str] = None) -> tuple[OCRPair, ...]:
    """List only the X5 PP-OCRv3 and S100 PP-OCRv6 manifest-backed pairs.

    ``auto`` is deliberately host-independent.  It lists both known pairs and
    does not inspect hardware; execution target authorization belongs to the
    real runner.
    """

    if target is None or str(target).strip().lower() == "auto":
        return tuple(_pair_for_target(value) for value in _SUPPORTED_PUBLISHED_TARGETS)
    key = _normalise_target(target)
    if key not in _SUPPORTED_PUBLISHED_TARGETS:
        return ()
    return (_pair_for_target(key),)


def resolve_pair(
    target: str = "auto",
    *,
    det_asset_id: Optional[str] = None,
    rec_asset_id: Optional[str] = None,
    det_model_path: Optional[str | Path] = None,
    rec_model_path: Optional[str | Path] = None,
) -> OCRPair:
    """Resolve one finite pair without loading SDKs or inspecting model bytes.

    Custom local paths are accepted only when both paths are associated with
    their exact qualified manifest references.  With ``target='auto'`` an
    explicit pair reference is required so selection remains host-independent.
    """

    if (det_model_path is None) != (rec_model_path is None):
        raise UnsupportedAssetError(
            "det_model_path and rec_model_path must be supplied together."
        )
    if (det_asset_id is None) != (rec_asset_id is None):
        raise UnsupportedAssetError(
            "det_asset_id and rec_asset_id must be supplied together."
        )
    if (det_model_path is not None) and det_asset_id is None:
        raise UnsupportedAssetError(
            "Custom model paths require exact det_asset_id and rec_asset_id references."
        )

    requested = _normalise_target(target, allow_auto=True)
    if requested == "auto":
        if det_asset_id is None or rec_asset_id is None:
            raise UnsupportedAssetError(
                "target='auto' needs both qualified asset references; use x5 or s100 "
                "to select the default pair."
            )
        candidates = [
            pair
            for pair in list_available_pairs()
            if pair.detector_asset == det_asset_id
            and pair.recognizer_asset == rec_asset_id
        ]
    else:
        if requested not in _SUPPORTED_PUBLISHED_TARGETS:
            raise UnsupportedAssetError(
                f"No audited PaddleOCR pair is published for target {requested!r}."
            )
        candidates = [pair for pair in list_available_pairs(requested)]
        if det_asset_id is not None:
            candidates = [
                pair
                for pair in candidates
                if pair.detector_asset == det_asset_id
                and pair.recognizer_asset == rec_asset_id
            ]

    if len(candidates) != 1:
        if det_asset_id is not None or rec_asset_id is not None:
            raise UnsupportedAssetError(
                "Detector and recognizer references are not one audited pair; "
                "mixed target/model-family pairs are rejected."
            )
        raise UnsupportedAssetError(
            f"No unique audited PaddleOCR pair for target={requested!r}."
        )

    selected = candidates[0]
    if det_model_path is not None:
        detector_path = Path(det_model_path).expanduser()
        recognizer_path = Path(rec_model_path).expanduser()  # type: ignore[arg-type]
        if detector_path.resolve() == recognizer_path.resolve():
            raise UnsupportedAssetError(
                "Detector and recognizer model paths must be distinct files."
            )
        selected = _with_paths(selected, detector_path, recognizer_path)
    return selected


def _pair_for_target(target: str) -> OCRPair:
    records = _manifest_records()
    if target == "x5":
        det = _find_record(records, "x5:paddleocr:en_PP-OCRv3_det_640x640_nv12.bin")
        rec = _find_record(records, "x5:paddleocr:en_PP-OCRv3_rec_48x320_rgb.bin")
        return _build_pair(
            target="x5",
            variant="ppocrv3",
            det=det,
            rec=rec,
            detector=StageContract(
                stage="detector",
                asset_id=det[0],
                target="x5",
                variant="ppocrv3",
                model_name="en_PP-OCRv3_det_infer-deploy_640x640_nv12",
                input_names=("x",),
                input_shapes={"x": (1, 3, 640, 640)},
                input_dtypes={"x": "nv12"},
                runtime_input_shapes={"x": (1, 960, 640, 1)},
                runtime_input_dtypes={"x": "uint8"},
                output_name="sigmoid_0.tmp_0",
                output_shape=(1, 1, 640, 640),
                output_dtype="float32",
                input_protocol="packed_nv12",
            ),
            recognizer=StageContract(
                stage="recognizer",
                asset_id=rec[0],
                target="x5",
                variant="ppocrv3",
                model_name="en_PP-OCRv3_rec_infer-deploy_48x320_rgb_NCHW",
                input_names=("x",),
                input_shapes={"x": (1, 3, 48, 320)},
                input_dtypes={"x": "float32"},
                runtime_input_shapes={"x": (1, 3, 48, 320)},
                runtime_input_dtypes={"x": "float32"},
                output_name="softmax_2.tmp_0",
                output_shape=(1, 40, 97, 1),
                output_dtype="float32",
                input_protocol="rgb_f32_nchw",
            ),
            vocabulary=VocabularySpec(kind="fixed", class_count=97),
        )
    if target == "s100":
        det = _find_record(
            records,
            "s:paddle_ocr:s100/PP-OCRv6_det_infer-deploy_640x640_nv12.hbm",
        )
        rec = _find_record(
            records,
            "s:paddle_ocr:s100/PP-OCRv6_rec_infer-deploy_48x320_rgb.hbm",
        )
        return _build_pair(
            target="s100",
            variant="ppocrv6",
            det=det,
            rec=rec,
            detector=StageContract(
                stage="detector",
                asset_id=det[0],
                target="s100",
                variant="ppocrv6",
                model_name="PP-OCRv6_det_infer-deploy_640x640_nv12",
                input_names=("x_y", "x_uv"),
                input_shapes={
                    "x_y": (1, 640, 640, 1),
                    "x_uv": (1, 320, 320, 2),
                },
                input_dtypes={"x_y": "uint8", "x_uv": "uint8"},
                runtime_input_shapes={
                    "x_y": (1, 640, 640, 1),
                    "x_uv": (1, 320, 320, 2),
                },
                runtime_input_dtypes={"x_y": "uint8", "x_uv": "uint8"},
                output_name="fetch_name_0",
                output_shape=(1, 1, 640, 640),
                output_dtype="float32",
                input_protocol="split_nv12",
            ),
            recognizer=StageContract(
                stage="recognizer",
                asset_id=rec[0],
                target="s100",
                variant="ppocrv6",
                model_name="PP-OCRv6_rec_infer-deploy_48x320_rgb",
                input_names=("x",),
                input_shapes={"x": (1, 3, 48, 320)},
                input_dtypes={"x": "float32"},
                runtime_input_shapes={"x": (1, 3, 48, 320)},
                runtime_input_dtypes={"x": "float32"},
                output_name="fetch_name_0",
                output_shape=(1, 40, 18710),
                output_dtype="float32",
                input_protocol="rgb_f32_nchw",
            ),
            vocabulary=VocabularySpec(
                kind="utf8_lines",
                class_count=18710,
                source_path=_TEST_DATA_DIR / "s100" / "ppocrv6_dict.txt",
                sha256=S100_VOCABULARY_SHA256,
            ),
        )
    raise UnsupportedAssetError(f"No audited pair for target {target!r}.")


def _build_pair(
    *,
    target: str,
    variant: str,
    det: tuple[str, str],
    rec: tuple[str, str],
    detector: StageContract,
    recognizer: StageContract,
    vocabulary: VocabularySpec,
) -> OCRPair:
    return OCRPair(
        target=target,
        variant=variant,
        detector_asset=det[0],
        recognizer_asset=rec[0],
        detector=detector,
        recognizer=recognizer,
        detector_model_path=_MODEL_DIR / Path(det[1]),
        recognizer_model_path=_MODEL_DIR / Path(rec[1]),
        vocabulary=vocabulary,
    )


def _with_paths(pair: OCRPair, det_path: Path, rec_path: Path) -> OCRPair:
    return OCRPair(
        target=pair.target,
        variant=pair.variant,
        detector_asset=pair.detector_asset,
        recognizer_asset=pair.recognizer_asset,
        detector=pair.detector,
        recognizer=pair.recognizer,
        detector_model_path=det_path,
        recognizer_model_path=rec_path,
        vocabulary=pair.vocabulary,
    )


def _manifest_records() -> tuple[tuple[str, str], ...]:
    """Return ``(qualified_reference, filename)`` from the shared reader."""

    try:
        from utils.py_utils.assets import list_assets
    except ImportError as exc:  # pragma: no cover - checkout-integrity guard
        raise BindingError("The shared manifest asset resolver is required.") from exc

    records: list[tuple[str, str]] = []
    for group, sample in (("x5", "paddleocr"), ("s", "paddle_ocr")):
        try:
            assets = list_assets(group, sample)
        except (OSError, KeyError, TypeError, ValueError) as exc:
            raise BindingError(f"Could not read {group}:{sample} manifest: {exc}") from exc
        for asset in assets:
            records.append((asset.reference, asset.filename))
    return tuple(records)


def _find_record(records: Sequence[tuple[str, str]], reference: str) -> tuple[str, str]:
    matches = [record for record in records if record[0] == reference]
    if len(matches) != 1:
        raise BindingError(f"Expected one manifest asset {reference!r}, found {len(matches)}.")
    return matches[0]


def _normalise_target(value: str, *, allow_auto: bool = False) -> str:
    key = (value or "").strip().lower()
    allowed = set(SUPPORTED_TARGETS)
    if allow_auto:
        allowed.add("auto")
    if key not in allowed:
        suffix = "/auto" if allow_auto else ""
        raise UnsupportedAssetError(
            f"Unknown target {value!r}; use x5/s100/s100p/s600{suffix}."
        )
    return key



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
        from utils.py_utils.platforms import resolve_target

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
    from utils.py_utils.assets import download_asset, resolve_asset

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
