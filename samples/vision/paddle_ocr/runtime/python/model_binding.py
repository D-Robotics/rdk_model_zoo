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

"""Finite PaddleOCR asset and runtime contracts.

The release manifests remain the authority for model asset facts.  This module
adds only the small, source-observed tensor contracts needed by the Python
composition pilot.  It does not import a board SDK, inspect model bytes, or
infer a protocol from a filename supplied by a caller.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from pathlib import Path
from typing import Any, Literal, Mapping, Optional, Sequence


SUPPORTED_TARGETS = ("x5", "s100", "s100p", "s600")
SUPPORTED_VARIANTS = ("ppocrv3", "ppocrv6")
_SUPPORTED_PUBLISHED_TARGETS = ("x5", "s100")
_SAMPLE_DIR = Path(__file__).resolve().parents[2]
_MODEL_DIR = _SAMPLE_DIR / "model"
_TEST_DATA_DIR = _SAMPLE_DIR / "test_data"

# This is the literal alphabet used by the legacy X5 PP-OCRv3 wrapper.  It
# intentionally contains two trailing spaces and therefore has 96 entries.
X5_ALPHABET = (
    "0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz"
    "{|}~!\"#$%&'()*+,-./  "
)
S100_VOCABULARY_SHA256 = (
    "b5f2bfe2bdd9448429e3e82b51c789775d9b42f2403d082b00662eb77e401c5d"
)


class BindingError(ValueError):
    """Base class for selection, metadata and tensor contract failures."""


class UnsupportedAssetError(BindingError):
    """The requested asset pair or target is outside the pilot boundary."""


class MetadataMismatchError(BindingError):
    """A loaded model or runner tensor does not satisfy its bound contract."""


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


# The name is useful to callers that prefer the more explicit wording.
ModelPair = OCRPair


@dataclass(frozen=True)
class RuntimeMetadata:
    """Facts read from one actual runtime model instance."""

    model_name: str
    input_names: tuple[str, ...]
    input_shapes: Mapping[str, tuple[int, ...]]
    output_names: tuple[str, ...]
    output_shapes: Mapping[str, tuple[int, ...]]
    input_dtypes: Mapping[str, str]
    output_dtypes: Mapping[str, str]
    input_strides: Mapping[str, tuple[int, ...]] = field(default_factory=dict)
    output_strides: Mapping[str, tuple[int, ...]] = field(default_factory=dict)
    output_scales: Mapping[str, Any] = field(default_factory=dict)
    output_zero_points: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "RuntimeMetadata":
        """Build metadata from flat or one-model nested runtime attributes."""

        model_name = str(values.get("model_name") or values.get("name") or "model")

        def model_field(name: str, default: Any) -> Any:
            value = values.get(name, default)
            if isinstance(value, Mapping) and model_name in value:
                return value[model_name]
            return value

        input_names = tuple(str(value) for value in model_field("input_names", ()))
        output_names = tuple(str(value) for value in model_field("output_names", ()))
        return cls(
            model_name=model_name,
            input_names=input_names,
            input_shapes=_normalise_shapes(model_field("input_shapes", {})),
            output_names=output_names,
            output_shapes=_normalise_shapes(model_field("output_shapes", {})),
            input_dtypes=_normalise_dtypes(model_field("input_dtypes", {})),
            output_dtypes=_normalise_dtypes(model_field("output_dtypes", {})),
            input_strides=_normalise_shapes(model_field("input_strides", {})),
            output_strides=_normalise_shapes(model_field("output_strides", {})),
            output_scales=_normalise_numbers(model_field("output_scales", {})),
            output_zero_points=_normalise_numbers(
                model_field("output_zero_points", {})
            ),
        )

    @classmethod
    def from_runtime(cls, runtime: Any) -> "RuntimeMetadata":
        """Read only public metadata attributes exposed by ``hbm_runtime``."""

        names = getattr(runtime, "model_names", None)
        if not names:
            raise MetadataMismatchError("Runtime did not expose model_names.")
        model_name = str(names[0])

        def runtime_field(name: str, default: Any) -> Any:
            value = getattr(runtime, name, default)
            if isinstance(value, Mapping) and model_name in value:
                return value[model_name]
            return value

        return cls.from_mapping(
            {
                "model_name": model_name,
                "input_names": runtime_field("input_names", ()),
                "input_shapes": runtime_field("input_shapes", {}),
                "output_names": runtime_field("output_names", ()),
                "output_shapes": runtime_field("output_shapes", {}),
                "input_dtypes": runtime_field("input_dtypes", {}),
                "output_dtypes": runtime_field("output_dtypes", {}),
                "input_strides": runtime_field("input_strides", {}),
                "output_strides": runtime_field("output_strides", {}),
                "output_scales": runtime_field("output_scales", {}),
                "output_zero_points": runtime_field("output_zero_points", {}),
            }
        )


@dataclass(frozen=True)
class StageBinding:
    """Static contract plus the exact metadata observed at runtime."""

    pair: OCRPair
    stage: Literal["detector", "recognizer"]
    contract: StageContract
    model_name: str
    input_names: tuple[str, ...]
    input_shapes: Mapping[str, tuple[int, ...]]
    input_dtypes: Mapping[str, str]
    output_name: str
    output_shape: tuple[int, ...]
    output_dtype: str
    runtime_metadata: RuntimeMetadata

    @property
    def runtime_input_shapes(self) -> Mapping[str, tuple[int, ...]]:
        """Return the physical tensor shapes used by the runtime runner."""

        return self.contract.runtime_input_shapes

    @property
    def runtime_input_dtypes(self) -> Mapping[str, str]:
        """Return the physical tensor dtypes used by the runtime runner."""

        return self.contract.runtime_input_dtypes


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


def bind_stage(
    pair: OCRPair,
    stage: Literal["detector", "recognizer"] | str,
    metadata: RuntimeMetadata | Mapping[str, Any],
) -> StageBinding:
    """Validate actual runtime metadata against the selected stage contract."""

    if stage not in ("detector", "recognizer"):
        raise BindingError(f"Unknown OCR stage {stage!r}.")
    contract = pair.detector if stage == "detector" else pair.recognizer
    facts = metadata if isinstance(metadata, RuntimeMetadata) else RuntimeMetadata.from_mapping(metadata)

    if facts.model_name != contract.model_name:
        raise MetadataMismatchError(
            f"{stage} model name {facts.model_name!r} does not match the audited "
            f"name {contract.model_name!r}."
        )
    if tuple(facts.input_names) != tuple(contract.input_names):
        raise MetadataMismatchError(
            f"{stage} input names {facts.input_names!r} do not match the audited "
            f"names {contract.input_names!r}."
        )
    if tuple(facts.output_names) != (contract.output_name,):
        raise MetadataMismatchError(
            f"{stage} output names {facts.output_names!r} do not match the audited "
            f"name {contract.output_name!r}."
        )
    for name in contract.input_names:
        actual_shape = facts.input_shapes.get(name)
        if tuple(actual_shape or ()) != tuple(contract.input_shapes[name]):
            raise MetadataMismatchError(
                f"{stage} input {name!r} shape {actual_shape!r} does not match "
                f"{contract.input_shapes[name]!r}."
            )
        actual_dtype = _canonical_dtype(facts.input_dtypes.get(name))
        if actual_dtype is None:
            raise MetadataMismatchError(
                f"{stage} input {name!r} metadata is missing a dtype."
            )
        if actual_dtype != contract.input_dtypes[name]:
            raise MetadataMismatchError(
                f"{stage} input {name!r} dtype {actual_dtype!r} does not match "
                f"the audited {contract.input_dtypes[name]!r} contract."
            )

    actual_output_shape = facts.output_shapes.get(contract.output_name)
    if tuple(actual_output_shape or ()) != tuple(contract.output_shape):
        raise MetadataMismatchError(
            f"{stage} output {contract.output_name!r} shape {actual_output_shape!r} "
            f"does not match {contract.output_shape!r}."
        )
    actual_output_dtype = _canonical_dtype(facts.output_dtypes.get(contract.output_name))
    if actual_output_dtype is None:
        raise MetadataMismatchError(
            f"{stage} output {contract.output_name!r} metadata is missing a dtype."
        )
    if actual_output_dtype != "float32":
        raise MetadataMismatchError(
            f"The OCR pilot accepts only F32 outputs; {stage} reported "
            f"{actual_output_dtype!r}."
        )
    if contract.output_name in facts.output_scales or contract.output_name in facts.output_zero_points:
        raise MetadataMismatchError(
            f"{stage} F32 output must not carry an unverified quantization mapping."
        )

    return StageBinding(
        pair=pair,
        stage=stage,  # type: ignore[arg-type]
        contract=contract,
        model_name=facts.model_name,
        input_names=facts.input_names,
        input_shapes=facts.input_shapes,
        input_dtypes=facts.input_dtypes,
        output_name=contract.output_name,
        output_shape=contract.output_shape,
        output_dtype=actual_output_dtype,
        runtime_metadata=facts,
    )


def validate_stage_inputs(
    binding: StageBinding | StageContract,
    inputs: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate and return a copy of one flat physical input mapping."""

    expected_names = tuple(binding.input_names)
    actual_names = tuple(inputs.keys())
    if set(actual_names) != set(expected_names) or len(actual_names) != len(expected_names):
        raise MetadataMismatchError(
            f"{binding.stage} runner inputs {actual_names!r} do not match "
            f"{expected_names!r}."
        )
    import numpy as np

    result: dict[str, Any] = {}
    for name in expected_names:
        value = np.asarray(inputs[name])
        expected_shape = binding.runtime_input_shapes[name]
        expected_dtype = np.dtype(binding.runtime_input_dtypes[name])
        if tuple(value.shape) != tuple(expected_shape):
            raise MetadataMismatchError(
                f"{binding.stage} input {name!r} runtime shape {value.shape!r} "
                f"does not match {expected_shape!r}."
            )
        if value.dtype != expected_dtype:
            raise MetadataMismatchError(
                f"{binding.stage} input {name!r} dtype {value.dtype!r} does not "
                f"match {expected_dtype!r}."
            )
        if not np.all(np.isfinite(value)):
            raise MetadataMismatchError(
                f"{binding.stage} input {name!r} contains NaN or infinity."
            )
        result[name] = value
    return result


def validate_stage_output(
    binding: StageBinding | StageContract,
    outputs: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate one flat F32 output mapping, including finite values."""

    import numpy as np

    if tuple(outputs.keys()) != (binding.output_name,):
        raise MetadataMismatchError(
            f"{binding.stage} runner outputs {tuple(outputs.keys())!r} do not "
            f"match {(binding.output_name,)!r}."
        )
    value = np.asarray(outputs[binding.output_name])
    if tuple(value.shape) != tuple(binding.output_shape):
        raise MetadataMismatchError(
            f"{binding.stage} output {binding.output_name!r} runtime shape "
            f"{value.shape!r} does not match {binding.output_shape!r}."
        )
    if value.dtype != np.dtype("float32"):
        raise MetadataMismatchError(
            f"{binding.stage} output {binding.output_name!r} dtype {value.dtype!r} "
            "does not match the F32 contract."
        )
    if not np.all(np.isfinite(value)):
        raise MetadataMismatchError(
            f"{binding.stage} output {binding.output_name!r} contains NaN or infinity."
        )
    return {binding.output_name: value}


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
        from samples._shared.assets import list_assets
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


def _normalise_shapes(values: Any) -> dict[str, tuple[int, ...]]:
    if not isinstance(values, Mapping):
        return {}
    result: dict[str, tuple[int, ...]] = {}
    for name, shape in values.items():
        try:
            result[str(name)] = tuple(int(dimension) for dimension in shape)
        except (TypeError, ValueError):
            result[str(name)] = ()
    return result


def _normalise_dtypes(values: Any) -> dict[str, str]:
    if not isinstance(values, Mapping):
        return {}
    result: dict[str, str] = {}
    for name, dtype in values.items():
        raw = str(getattr(dtype, "name", dtype)).lower()
        if raw in {"f32", "float", "float32", "hbdnndatatype.f32"} or raw.endswith(".f32"):
            raw = "float32"
        elif raw in {"u8", "uint8", "hbdnndatatype.u8"} or raw.endswith(".u8"):
            raw = "uint8"
        elif raw in {"nv12", "hbdnndatatype.nv12"} or raw.endswith(".nv12"):
            raw = "nv12"
        result[str(name)] = raw
    return result


def _normalise_numbers(values: Any) -> dict[str, Any]:
    if not isinstance(values, Mapping):
        return {}
    result: dict[str, Any] = {}
    for name, value in values.items():
        shape = getattr(value, "shape", None)
        if isinstance(value, (list, tuple, Mapping)) or (
            shape is not None and tuple(shape) != ()
        ):
            # Keep vectors/structured values present so the finite F32
            # contract rejects them instead of dropping unsupported metadata.
            result[str(name)] = value
            continue
        try:
            result[str(name)] = float(value)
        except (OverflowError, TypeError, ValueError):
            # Preserve an unnormalisable but present value.  F32 OCR output
            # does not support quantization metadata; bind_stage must see the
            # key and reject it rather than silently treating it as absent.
            result[str(name)] = value
    return result


def _canonical_dtype(dtype: Any) -> Optional[str]:
    if dtype is None:
        return None
    return _normalise_dtypes({"dtype": dtype}).get("dtype")


__all__ = [
    "BindingError",
    "MetadataMismatchError",
    "ModelPair",
    "OCRPair",
    "RuntimeMetadata",
    "S100_VOCABULARY_SHA256",
    "SUPPORTED_TARGETS",
    "SUPPORTED_VARIANTS",
    "StageBinding",
    "StageContract",
    "UnsupportedAssetError",
    "VocabularySpec",
    "X5_ALPHABET",
    "bind_stage",
    "list_available_pairs",
    "resolve_pair",
    "validate_stage_inputs",
    "validate_stage_output",
]
