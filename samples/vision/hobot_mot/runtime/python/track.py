"""HobotMot (IOU 2.0) tracker model: detections in, track identities out.

The public interface follows the sample runtime contract: ``preprocess`` ->
``infer`` -> ``postprocess`` chained by ``predict``. ``infer`` calls the same
genuine upstream HobotMot library as the C++ runtime through the sample's C
ABI; no Python re-implementation of the algorithm exists here.

Stage contract:

* Input: one frame of :class:`Detection` (finite xyxy pixel coordinates,
  score in [0, 1], integer class id) plus a millisecond timestamp that must
  strictly increase per instance.
* Tensors: there is no neural model; the physical backend mapping is the C
  ABI buffer layout (n*4 float boxes, n scores, n int32 class ids).
* Context: the frame's pixel size and timestamp, validated per call.
* RawOutputs: per-detection upstream rows (tracker box, id, state) and the
  identities the tracker removed in this frame.
* Result: :class:`FrameTracks` aligned with the input order.
"""
from __future__ import annotations

import json
import math
import struct
from dataclasses import dataclass, field
from pathlib import Path

from native import (HobotMotNative, MOT_CAPI_ERR_CONFIG, MotNativeError)

#: Sample root (runtime/python/track.py -> sample directory).
SAMPLE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = SAMPLE_ROOT / "model" / "configs" / "iou2.json"

STATE_VALID = 0
STATE_INVALID = 4
STATE_NAMES = {0: "valid", 1: "filtered", 2: "invisible", 3: "disappeared",
               4: "invalid"}


class MotConfigError(ValueError):
    """Tracker configuration rejected before any native call."""


class MotInputError(ValueError):
    """Frame input rejected before any native call."""


@dataclass
class Detection:
    """One input detection in pixel coordinates (x1, y1)-(x2, y2)."""

    bbox: tuple[float, float, float, float]
    score: float
    class_id: int


@dataclass
class TrackResult:
    """One output row, aligned with the input detection at the same index."""

    bbox: tuple[int, int, int, int]
    score: float
    class_id: int
    track_id: int  # -1 when state is not valid
    state: str


@dataclass
class DisappearedId:
    """A track identity the tracker removed in this frame."""

    class_id: int
    track_id: int


@dataclass
class PreparedFrame:
    """Input plus the geometry of this call (the call's Context)."""

    detections: list[Detection]
    timestamp_ms: int
    frame_width: int
    frame_height: int


@dataclass
class RawMotFrame:
    """RawOutputs: per-detection rows and removed ids, input order kept."""

    ids: list[int]
    boxes: list[tuple[int, int, int, int]]
    states: list[int]
    disappeared: list[tuple[int, int]]  # (class_id, track_id)


@dataclass
class FrameTracks:
    """Result of one frame."""

    timestamp_ms: int
    tracks: list[TrackResult] = field(default_factory=list)
    disappeared: list[DisappearedId] = field(default_factory=list)


# JSON type each upstream IOU2Param field actually reads (see upstream
# include/data_type/mot_data_type.hpp): mistyped fields keep upstream
# defaults silently, so they are rejected here instead. Integer ranges
# mirror the upstream C++ members (jsoncpp isInt/isUInt would silently
# reject out-of-range integers, keeping the default). kind: b=bool, i=int,
# u=uint, d=number, s=string.
_INT32_MAX = 2147483647
_INT32_MIN = -2147483648
_UINT32_MAX = 4294967295
_INT64_MAX = 9223372036854775807
# Coordinates must stay inside this magnitude so the float32 conversion of
# the C ABI and the upstream int conversion are both exact and defined.
_COORD_MAX = 2 ** 30

_FIELDS = {
    "match_type": ("s", (0.0, 0.0)),
    "use_kalman_filter": ("b", (0.0, 0.0)),
    "missing_time_thres": ("i", (0, _INT32_MAX)),
    "vanish_frame_count": ("u", (0, _UINT32_MAX)),
    "time_gap": ("u", (1, _UINT32_MAX)),
    "remove_obsolete_track": ("i", (0, _INT32_MAX)),
    "iou_thres": ("d", (0.0, 1.0)),
    "euclidean_thres": ("d", (0.0, None)),
    "use_location_gain": ("i", (0, _INT32_MAX)),
    "max_trajectory_number": ("i", (1, _INT32_MAX)),
    "min_score": ("d", (0.0, 1.0)),
    "ignore_overlap_thres": ("d", (0.0, 1.0)),
}

_KIND_NAMES = {"b": "bool", "i": "int", "u": "unsigned int", "d": "number",
               "s": "string"}


def _is_plain_int(value) -> bool:
    """True for real integers; JSON-style bools never qualify."""
    return isinstance(value, int) and not isinstance(value, bool)


def _to_float32(value: float) -> float:
    """Round to the float32 value the native tracker keeps."""
    return struct.unpack("f", struct.pack("f", value))[0]


def _dimension_error(name: str, value) -> str:
    return f"{name} must be an integer in (0, {_INT32_MAX}], got {value!r}"


def _resolve_dimension(value, default: int, name: str) -> int:
    """Pick an explicit frame dimension or the instance default.

    ``None`` and the integer 0 mean "unspecified" (the C++ runtime uses 0
    as its marker); any other value must be a real integer in range —
    ``int()`` coercion would silently accept ``True``, ``1.5`` and
    ``"100"``.
    """
    if value is None or value == 0:
        return default
    if not _is_plain_int(value) or not 0 < value <= _INT32_MAX:
        raise MotInputError(_dimension_error(name, value))
    return value


def validate_config(config: dict, source: str = "config") -> None:
    """Validate parsed tracker config against the upstream IOU2 contract.

    Args:
        config: parsed JSON object.
        source: path or label used in error messages.

    Raises:
        MotConfigError: on non-object JSON, missing/unsupported
            ``tracker_type``, or a field whose JSON type or range would make
            upstream silently keep its default.
    """
    if not isinstance(config, dict):
        raise MotConfigError(f"{source}: config must be a JSON object")
    tracker_type = config.get("tracker_type")
    if tracker_type != "IOU_2.0":
        raise MotConfigError(
            f"{source}: 'tracker_type' must be the string \"IOU_2.0\" (the "
            f"only tracker compiled in the pinned upstream), got "
            f"{tracker_type!r}")
    for name, (kind, bounds) in _FIELDS.items():
        if name not in config:
            continue
        value = config[name]
        if kind == "b":
            ok = isinstance(value, bool)
        elif kind in ("i", "u"):
            ok = _is_plain_int(value)
        elif kind == "d":
            ok = isinstance(value, (int, float)) and not isinstance(value, bool)
        else:
            ok = isinstance(value, str)
        if not ok:
            raise MotConfigError(
                f"{source}: field '{name}' must be JSON "
                f"{_KIND_NAMES[kind]}; upstream would silently ignore this "
                f"value")
        if kind == "s":
            if name == "match_type" and value not in ("IOU", "Euclidean"):
                raise MotConfigError(
                    f"{source}: 'match_type' must be \"IOU\" or "
                    f"\"Euclidean\", got {value!r}")
            continue
        if kind == "b":
            continue
        if not math.isfinite(value):
            raise MotConfigError(f"{source}: field '{name}' must be finite")
        low, high = bounds
        if value < low or (high is not None and value > high):
            upper = high if high is not None else "inf"
            raise MotConfigError(
                f"{source}: field '{name}' = {value} out of range "
                f"[{low}, {upper}]")


def validate_config_file(config_path: str | Path) -> dict:
    """Load, parse and validate a tracker config file.

    Raises:
        MotConfigError: when the file is missing, unreadable, invalid JSON
            or violates :func:`validate_config`.
    """
    path = Path(config_path)
    try:
        with path.open(encoding="utf-8") as handle:
            config = json.load(handle)
    except OSError as exc:
        raise MotConfigError(f"cannot open tracker config: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise MotConfigError(
            f"{path}: tracker config is not valid JSON: {exc}") from exc
    validate_config(config, str(path))
    return config


class HobotMotTracker:
    """Multi-object tracker over detection sequences (one per stream).

    The upstream tracker is class-agnostic, so one native instance is kept
    per class id and track identities are unique only inside their class
    namespace (``reset`` restarts all of them).

    Construction validates the config (see :func:`validate_config_file`) and
    locates the native library; both raise before any upstream object is
    created.
    """

    def __init__(self, config_path: str | Path | None = None,
                 frame_width: int = 960, frame_height: int = 544,
                 native_lib: str | Path | None = None):
        # The instance defaults themselves must be exact integers: they feed
        # the int32 fields of the C ABI. Checked before anything is opened
        # or loaded.
        for name, value in (("frame_width", frame_width),
                            ("frame_height", frame_height)):
            if not _is_plain_int(value) or not 0 < value <= _INT32_MAX:
                raise MotConfigError(_dimension_error(name, value))
        self._config_path = str(config_path or DEFAULT_CONFIG)
        self._frame_width = frame_width
        self._frame_height = frame_height
        validate_config_file(self._config_path)
        self._native = HobotMotNative(self._config_path, self._frame_width,
                                      self._frame_height, native_lib)
        self._last_timestamp_ms = -1

    def preprocess(self, detections, timestamp_ms: int,
                   frame_width: int | None = None,
                   frame_height: int | None = None) -> PreparedFrame:
        """Validate one frame and its geometry.

        Raises:
            MotInputError: on non-Detection rows, non-finite or
                non-int32-representable coordinates, zero or negative box
                extents, score outside [0, 1], class ids outside int32,
                non-integer frame size or a timestamp that does not
                strictly increase.
        """
        width = _resolve_dimension(frame_width, self._frame_width,
                                   "frame_width")
        height = _resolve_dimension(frame_height, self._frame_height,
                                    "frame_height")
        checked: list[Detection] = []
        for index, det in enumerate(detections):
            if not isinstance(det, Detection):
                raise MotInputError(
                    f"detection {index} is not a Detection, got {type(det)!r}")
            checked.append(det)
        prepared = PreparedFrame(checked, timestamp_ms, width, height)
        self._validate_frame(prepared)
        return prepared

    def _validate_frame(self, prepared: PreparedFrame) -> None:
        """Shared input gate for preprocess and the public infer.

        Every value that crosses the C ABI as an integer is checked against
        the width ctypes actually marshals (int32/int64), so nothing wraps
        silently inside the native call.
        """
        if not _is_plain_int(prepared.timestamp_ms) \
                or not 0 <= prepared.timestamp_ms <= _INT64_MAX:
            raise MotInputError(
                f"timestamp_ms must be an integer in [0, {_INT64_MAX}], got "
                f"{prepared.timestamp_ms!r}")
        if not _is_plain_int(prepared.frame_width) \
                or not 0 < prepared.frame_width <= _INT32_MAX:
            raise MotInputError(
                f"frame_width must be an integer in (0, {_INT32_MAX}]")
        if not _is_plain_int(prepared.frame_height) \
                or not 0 < prepared.frame_height <= _INT32_MAX:
            raise MotInputError(
                f"frame_height must be an integer in (0, {_INT32_MAX}]")
        if prepared.timestamp_ms <= self._last_timestamp_ms:
            raise MotInputError(
                f"timestamp_ms must strictly increase: previous "
                f"{self._last_timestamp_ms}, got {prepared.timestamp_ms}")
        for index, det in enumerate(prepared.detections):
            if len(det.bbox) != 4:
                raise MotInputError(
                    f"detection {index}: bbox must hold 4 values")
            if not all(isinstance(v, (int, float))
                       and not isinstance(v, bool) and math.isfinite(v)
                       and -_COORD_MAX <= v <= _COORD_MAX
                       for v in det.bbox):
                raise MotInputError(
                    f"detection {index}: coordinates must be finite numbers "
                    f"within ±{_COORD_MAX}")
            x1, y1, x2, y2 = det.bbox
            if x2 <= x1 or y2 <= y1:
                raise MotInputError(
                    f"detection {index}: box extents must be positive "
                    f"(x2 > x1 and y2 > y1), got {det.bbox}")
            if not isinstance(det.score, (int, float)) \
                    or isinstance(det.score, bool) \
                    or not math.isfinite(det.score) \
                    or not 0.0 <= det.score <= 1.0:
                raise MotInputError(
                    f"detection {index}: score must be a number within "
                    f"[0, 1], got {det.score!r}")
            if not _is_plain_int(det.class_id) \
                    or not _INT32_MIN <= det.class_id <= _INT32_MAX:
                raise MotInputError(
                    f"detection {index}: class_id must be an integer within "
                    f"int32 range, got {det.class_id!r}")

    def infer(self, prepared: PreparedFrame) -> RawMotFrame:
        """Run the upstream tracker for this frame through the C ABI.

        ``infer`` re-runs the full input gate, so staged calls cannot bypass
        :meth:`preprocess` validation or replay a stale timestamp.
        """
        self._validate_frame(prepared)
        boxes: list[float] = []
        scores: list[float] = []
        class_ids: list[int] = []
        for det in prepared.detections:
            boxes.extend(float(v) for v in det.bbox)
            scores.append(float(det.score))
            class_ids.append(int(det.class_id))
        try:
            ids, out_boxes, states, disappeared = self._native.process(
                boxes, scores, class_ids, prepared.timestamp_ms,
                prepared.frame_width, prepared.frame_height)
        except MotNativeError as exc:
            # Keep the C++ boundary: a config the file changed into after
            # construction is a config problem, everything else is input.
            if exc.code == MOT_CAPI_ERR_CONFIG:
                raise MotConfigError(str(exc)) from exc
            raise MotInputError(str(exc)) from exc
        self._last_timestamp_ms = prepared.timestamp_ms
        return RawMotFrame(ids, out_boxes, states, disappeared)

    def postprocess(self, raw: RawMotFrame,
                    prepared: PreparedFrame) -> FrameTracks:
        """Merge raw upstream rows into input-aligned results."""
        if len(raw.ids) != len(prepared.detections) \
                or len(raw.boxes) != len(prepared.detections) \
                or len(raw.states) != len(prepared.detections):
            raise MotInputError(
                f"raw output holds {len(raw.ids)} rows for "
                f"{len(prepared.detections)} detections")
        tracks = [
            TrackResult(
                bbox=tuple(raw.boxes[i]),
                # The C ABI transports float32; mirror that rounding so
                # Python results equal the C++ runtime bit for bit.
                score=_to_float32(prepared.detections[i].score),
                class_id=prepared.detections[i].class_id,
                track_id=raw.ids[i],
                state=STATE_NAMES.get(raw.states[i], "other"))
            for i in range(len(raw.ids))
        ]
        disappeared = [DisappearedId(int(cls), int(tid))
                       for cls, tid in raw.disappeared]
        return FrameTracks(prepared.timestamp_ms, tracks, disappeared)

    def predict(self, detections, timestamp_ms: int,
                frame_width: int | None = None,
                frame_height: int | None = None) -> FrameTracks:
        """preprocess -> infer -> postprocess for one frame."""
        prepared = self.preprocess(detections, timestamp_ms, frame_width,
                                   frame_height)
        raw = self.infer(prepared)
        return self.postprocess(raw, prepared)

    def reset(self) -> None:
        """Drop all tracker state; track ids restart from 1.

        Upstream has no in-place reset: this rebuilds every per-class
        instance with the same config, exactly like a new tracker.
        """
        self._native.reset()
        self._last_timestamp_ms = -1

    def close(self) -> None:
        """Release the native instance (idempotent)."""
        self._native.close()

    def __enter__(self) -> "HobotMotTracker":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()
