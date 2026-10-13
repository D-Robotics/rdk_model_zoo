"""ctypes binding for libhobot_mot_capi.so.

This module owns the complete Python side of the sample's C ABI (see
runtime/cpp/inc/mot_c_api.h): library discovery, argument marshalling and
error translation. The native library is only loaded when a tracker is
constructed, so ``--help``, argument validation and config validation work on
hosts without the ARM64 build.
"""
from __future__ import annotations

import ctypes
import os
from pathlib import Path

#: Sample root (runtime/python/native.py -> sample directory).
SAMPLE_ROOT = Path(__file__).resolve().parents[2]

_ERROR_BUFFER_SIZE = 512

MOT_CAPI_OK = 0
MOT_CAPI_ERR_PARAM = 1
MOT_CAPI_ERR_CONFIG = 2
MOT_CAPI_ERR_INPUT = 3
MOT_CAPI_ERR_STATE = 4
MOT_CAPI_ERR_CAPACITY = 5
MOT_CAPI_ERR_INTERNAL = 6

STATE_VALID = 0
STATE_INVALID = 4


class MotNativeError(RuntimeError):
    """Raised when the native tracker reports a failure.

    ``code`` carries the MOT_CAPI_* error code so callers can distinguish
    config failures (retryable after fixing the file) from input failures.
    """

    def __init__(self, message: str, code: int | None = None):
        super().__init__(message)
        self.code = code


def find_native_library(explicit: str | os.PathLike | None = None) -> Path:
    """Return the path of the C ABI shared library.

    Order: explicit argument, ``HOBOT_MOT_NATIVE_LIB`` environment variable,
    then any ``runtime/cpp/build/<target>/libhobot_mot_capi.so`` produced by
    the documented build step.

    Raises:
        MotNativeError: when no library can be located.
    """
    if explicit:
        path = Path(explicit).expanduser()
        if not path.exists():
            raise MotNativeError(f"native library not found: {path}")
        return path
    env = os.environ.get("HOBOT_MOT_NATIVE_LIB")
    if env:
        path = Path(env).expanduser()
        if not path.exists():
            raise MotNativeError(f"HOBOT_MOT_NATIVE_LIB does not exist: {path}")
        return path
    build_dir = SAMPLE_ROOT / "runtime/cpp/build"
    candidates = [build_dir / "libhobot_mot_capi.so"]
    candidates += sorted(build_dir.glob("*/libhobot_mot_capi.so"))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise MotNativeError(
        "libhobot_mot_capi.so not found; build it on the board with "
        "`cmake -B runtime/cpp/build -S runtime/cpp && cmake --build "
        "runtime/cpp/build` (see runtime/cpp/README.md), or pass "
        "--native-lib / HOBOT_MOT_NATIVE_LIB"
    )


class HobotMotNative:
    """One native tracker instance (one video stream).

    Wraps a ``mot_capi_handle_t``; destruction goes through ``close()`` and
    the context manager. Not thread-safe: one instance per stream, calls in
    frame order.
    """

    def __init__(self, config_path: str, frame_width: int = 960,
                 frame_height: int = 544, library: str | os.PathLike | None = None):
        self._lib = ctypes.CDLL(str(find_native_library(library)))
        self._declare()
        self._handle = ctypes.c_void_p()
        err = ctypes.create_string_buffer(_ERROR_BUFFER_SIZE)
        code = self._lib.mot_capi_create(
            config_path.encode(), frame_width, frame_height,
            ctypes.byref(self._handle), err, _ERROR_BUFFER_SIZE)
        self._check(code, err, "mot_capi_create")

    def _declare(self) -> None:
        lib = self._lib
        lib.mot_capi_create.argtypes = [
            ctypes.c_char_p, ctypes.c_int32, ctypes.c_int32,
            ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p, ctypes.c_size_t]
        lib.mot_capi_create.restype = ctypes.c_int
        lib.mot_capi_destroy.argtypes = [ctypes.c_void_p]
        lib.mot_capi_destroy.restype = ctypes.c_int
        lib.mot_capi_process.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_int32),
            ctypes.c_int32, ctypes.c_int64, ctypes.c_int32, ctypes.c_int32,
            ctypes.c_char_p, ctypes.c_size_t]
        lib.mot_capi_process.restype = ctypes.c_int
        lib.mot_capi_result_count.argtypes = [ctypes.c_void_p]
        lib.mot_capi_result_count.restype = ctypes.c_int32
        lib.mot_capi_copy_results.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32),
            ctypes.c_int32, ctypes.c_char_p, ctypes.c_size_t]
        lib.mot_capi_copy_results.restype = ctypes.c_int
        lib.mot_capi_disappeared_count.argtypes = [ctypes.c_void_p]
        lib.mot_capi_disappeared_count.restype = ctypes.c_int32
        lib.mot_capi_copy_disappeared.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int32), ctypes.c_int32, ctypes.c_char_p,
            ctypes.c_size_t]
        lib.mot_capi_copy_disappeared.restype = ctypes.c_int
        lib.mot_capi_reset.argtypes = [
            ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t]
        lib.mot_capi_reset.restype = ctypes.c_int

    @staticmethod
    def _check(code: int, err: ctypes.Array, call: str) -> None:
        if code != MOT_CAPI_OK:
            message = err.value.decode(errors="replace") or f"code {code}"
            raise MotNativeError(f"{call} failed: {message}", code)

    def process(self, boxes, scores, class_ids, timestamp_ms: int,
                frame_width: int, frame_height: int):
        """Run one frame; return ``(ids, boxes_out, states, disappeared)``.

        Args:
            boxes: flat sequence of n*4 floats (x1, y1, x2, y2).
            scores: n floats.
            class_ids: n ints.
            timestamp_ms: strictly increasing per instance.
            frame_width/frame_height: positive pixel size of this frame.
        """
        n = len(scores)
        if len(class_ids) != n or len(boxes) != 4 * n:
            raise MotNativeError("boxes/scores/class_ids length mismatch")
        c_boxes = (ctypes.c_float * max(1, 4 * n))(
            *[float(v) for v in boxes]) if n else None
        c_scores = (ctypes.c_float * max(1, n))(
            *[float(v) for v in scores]) if n else None
        c_classes = (ctypes.c_int32 * max(1, n))(
            *[int(v) for v in class_ids]) if n else None
        err = ctypes.create_string_buffer(_ERROR_BUFFER_SIZE)
        code = self._lib.mot_capi_process(
            self._handle, c_boxes, c_scores, c_classes, n,
            int(timestamp_ms), int(frame_width), int(frame_height),
            err, _ERROR_BUFFER_SIZE)
        self._check(code, err, "mot_capi_process")

        count = self._lib.mot_capi_result_count(self._handle)
        out_boxes = (ctypes.c_int32 * max(1, 4 * count))()
        out_ids = (ctypes.c_int32 * max(1, count))()
        out_states = (ctypes.c_int32 * max(1, count))()
        if count:
            code = self._lib.mot_capi_copy_results(
                self._handle, out_boxes, out_ids, out_states, count,
                err, _ERROR_BUFFER_SIZE)
            self._check(code, err, "mot_capi_copy_results")
        gone_count = self._lib.mot_capi_disappeared_count(self._handle)
        gone_classes = (ctypes.c_int32 * max(1, gone_count))()
        gone_ids = (ctypes.c_int32 * max(1, gone_count))()
        if gone_count:
            code = self._lib.mot_capi_copy_disappeared(
                self._handle, gone_classes, gone_ids, gone_count,
                err, _ERROR_BUFFER_SIZE)
            self._check(code, err, "mot_capi_copy_disappeared")

        boxes_out = [tuple(out_boxes[4 * i:4 * i + 4]) for i in range(count)]
        return (list(out_ids[:count]), boxes_out, list(out_states[:count]),
                [(gone_classes[i], gone_ids[i]) for i in range(gone_count)])

    def reset(self) -> None:
        err = ctypes.create_string_buffer(_ERROR_BUFFER_SIZE)
        code = self._lib.mot_capi_reset(self._handle, err, _ERROR_BUFFER_SIZE)
        self._check(code, err, "mot_capi_reset")

    def close(self) -> None:
        if getattr(self, "_handle", None) is not None and self._handle:
            self._lib.mot_capi_destroy(self._handle)
            self._handle = ctypes.c_void_p()

    def __enter__(self) -> "HobotMotNative":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()
