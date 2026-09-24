#!/usr/bin/env python3
# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Compare a fixed-source C++ capture against a unified C++ dump, on-board.

Reads the instrumented fixed-source capture (observer ``capture.json`` plus
the external runner's ``run-record.json`` from run_capture.py) and the unified
binary's dump (``manifest.json`` from ``yolov5_cpp --dump-dir``) from the SAME
board, restores logical arrays from each side's recorded physical layout, and
applies the fixed comparison criteria below. Each side ran its own complete
pipeline; no result of one side is ever fed to the other.

Criteria (fixed; never adjusted per run):
  inputs          exact byte equality of the logical (stride-restored) arrays,
                  dtypes and shapes must match on both sides
  raw outputs     allclose atol=1e-5, rtol=0 (int32 raw compares exactly),
                  dtypes must match; float values must be finite
  scale/zero      exact float32-BIT equality (the native quantization is
                  float32; decimal spellings of the same float32 compare equal,
                  different bits do not — no widened tolerance)
  thresholds      float32-bit equality (0.45f vs "0.450000" is the same bits)
  boxes (model space)   allclose atol=1e-4, rtol=0
  scores          allclose atol=1e-5, rtol=0
  class ids       exact
  boxes (final image space) required on BOTH sides — a missing or one-sided
                  capture blocks acceptance; there is no "pass without final
                  coordinates" mode
  Detection order on both sides is normalized by sorting
  (class_id, -score, x1, y1, x2, y2) — declared in every report.

Identity rules: the runner's run-record.json is REQUIRED and cross-checked —
its real subprocess return code must be 0, its binary/model/image pre/post
hashes must be present, equal before and after, equal to the observer's
capture-time hashes and (for binaries) to the hashes of the binaries this
comparison attests; the instrumentation audit verification must have passed;
board identity (runner + capture) must not conflict with the target. Unified
manifest model/image/binary hashes are REQUIRED and must match the same
bytes. Every payload on both sides is verified against the size and SHA-256
recorded at write time. Structural rules: the target-appropriate input count,
exactly three uniquely-shaped output heads and a complete shape bijection,
matching dtypes and quantization descriptors on both sides — empty, short,
duplicated or extra lists fail. The source ``detections`` and
``detections_original`` keys are required (a missing key never passes as an
observed empty list).

Every failure path — including missing files, JSON errors and unexpected
exceptions — still writes a complete failure report and exits nonzero.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import pathlib
import shutil
import sys
import traceback

import numpy as np

CRITERIA = {
    "inputs": {"mode": "exact"},
    "raw": {"mode": "allclose", "atol": 1e-5, "rtol": 0.0},
    "boxes": {"mode": "allclose", "atol": 1e-4, "rtol": 0.0},
    "scores": {"mode": "allclose", "atol": 1e-5, "rtol": 0.0},
    "class_ids": {"mode": "exact"},
}
EXPECTED_INPUTS = {"x5": 1, "s100": 2, "s100p": 2, "s600": 2}
EXPECTED_OUTPUTS = 3
KNOWN_DTYPES = {"uint8", "int32", "float32"}
DTYPES = {"uint8": np.uint8, "int32": np.int32, "float32": np.float32}
# Exact board-identity aliases from the repository registry
# (docs/release/platforms.json, schema 1) — the same contract
# samples/_shared/platforms.py::match_target applies. Prefix matching is
# NOT identity: S100P is a distinct target from s100 and unknown strings
# never pass.
TARGET_ALIASES = {
    "x5": {"x5"},
    "s100": {"s100"},
    "s100p": {"s100p"},
    "s600": {"s600"},
}
# socinfo names resolve to their target (X5 boards report X5U there and have
# no boardinfo soc_name at all).
TARGET_SOCINFO_ALIASES = {
    "x5": {"x5", "x5u", "x5h", "x5m"},
    "s100": {"s100"},
    "s100p": {"s100p"},
    "s600": {"s600"},
}


class ComparisonError(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 16), b""):
            digest.update(block)
    return digest.hexdigest()


def f32_bits(value) -> bytes:
    return np.float32(value).tobytes()


def f32_equal(a, b) -> bool:
    return f32_bits(a) == f32_bits(b)


def sanitize_stride(stride) -> list[int]:
    if not stride:
        return [0, 0, 0, 0]
    return [0 if value is None else int(value) for value in stride]


def restore_logical(raw: bytes, dtype: str, shape: list[int], stride: list[int]) -> np.ndarray:
    """Restores the NHWC logical array the addressing formula reads.

    Only two layout families are accepted — anything else is rejected rather
    than decoded on a guess:
      * strided: stride[2] > 0 and stride[3] > 0, stride[1] == width*stride[2]
        (the uniform row pitch both native readers rely on); stride[0] must be
        positive and cover the plane;
      * compact: all four strides zero/null AND the payload is exactly
        count*itemsize bytes (a longer payload is not a proven compact layout).
    """
    element = DTYPES[dtype]
    item = np.dtype(element).itemsize
    n, h, w, c = (int(v) for v in shape)
    if n != 1:
        raise ComparisonError(f"batch shape {shape} is not supported")
    count = h * w * c
    s0, s1, s2, s3 = (int(v) for v in stride)
    if s2 > 0 and s3 > 0:
        if s3 < item or s3 % item != 0:
            raise ComparisonError(
                f"channel stride {s3} overlaps or misaligns {dtype} elements "
                f"(itemsize {item}); channel stride must be >= itemsize and a "
                "multiple of it")
        if s2 < c * s3:
            raise ComparisonError(
                f"pixel stride {s2} < channels*stride[3] ({c} * {s3}): "
                "overlapping pixels — refusing to decode")
        if s1 != w * s2:
            raise ComparisonError(
                f"strided layout with stride[1]={s1} != width*stride[2]={w * s2} "
                "is not a proven layout — refusing to decode")
        if s0 <= 0 or s0 < h * s1:
            raise ComparisonError(f"stride[0]={s0} does not cover the plane")
        if (h - 1) * s1 + (w - 1) * s2 + (c - 1) * s3 + item > len(raw):
            raise ComparisonError(f"payload of {len(raw)} bytes does not cover "
                                  f"shape {shape} at strides {stride}")
        offsets = (np.arange(h, dtype=np.int64)[:, None, None] * (w * s2) +
                   np.arange(w, dtype=np.int64)[None, :, None] * s2 +
                   np.arange(c, dtype=np.int64)[None, None, :] * s3)
        buffer = np.frombuffer(raw, dtype=np.uint8)
        gathered = buffer[offsets[..., None] + np.arange(item, dtype=np.int64)]
        return gathered.reshape(h, w, c, item).copy().view(element).reshape(-1)
    if s0 == 0 and s1 == 0 and s2 == 0 and s3 == 0:
        expected = count * item
        if len(raw) != expected:
            raise ComparisonError(
                f"compact layout requires exactly {expected} bytes for shape "
                f"{shape}; payload has {len(raw)} — refusing to decode on a guess")
        return np.frombuffer(raw[:expected], dtype=element)
    raise ComparisonError(
        f"unsupported mixed strides {stride} for shape {shape} — refusing")


def restore_rows(raw: bytes, shape: list[int], stride: list[int]) -> bytes:
    """Gathers the deterministic NV12 plane rows out of a strided payload."""
    _, rows, cols, channels = (int(v) for v in shape)
    row_bytes = cols * channels
    pitch = int(stride[1])
    if pitch < row_bytes:
        raise ComparisonError(
            f"plane pitch {pitch} < row bytes {row_bytes}: overlapping rows")
    needed = (rows - 1) * pitch + row_bytes
    if len(raw) < needed:
        raise ComparisonError(
            f"plane payload of {len(raw)} bytes < required {needed}")
    out = bytearray()
    for row in range(rows):
        start = row * pitch
        out += raw[start:start + row_bytes]
    return bytes(out)


def load_json(path: pathlib.Path, what: str) -> dict:
    if not path.is_file():
        raise ComparisonError(f"missing {what}: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ComparisonError(f"unreadable {what} at {path}: {error}") from error


def stage_report(name: str, ok: bool, detail: str) -> dict:
    return {"stage": name, "passed": ok, "detail": detail}


def compare_arrays(a: np.ndarray, b: np.ndarray, rule: dict,
                   require_same_dtype: bool = False) -> tuple[bool, str]:
    if a.shape != b.shape:
        return False, f"shape {a.shape} vs {b.shape}"
    if require_same_dtype and a.dtype != b.dtype:
        return False, f"dtype {a.dtype} vs {b.dtype} (no silent widening)"
    if rule["mode"] == "exact":
        ok = bool(np.array_equal(a, b))
        detail = "exact" if ok else "values differ"
    else:
        if a.dtype.kind == "f" and not np.isfinite(a).all():
            return False, "source side contains non-finite values"
        if b.dtype.kind == "f" and not np.isfinite(b).all():
            return False, "unified side contains non-finite values"
        ok = bool(np.allclose(a.astype(np.float64), b.astype(np.float64),
                              atol=rule["atol"], rtol=rule["rtol"]))
        if not ok:
            diff = np.abs(a.astype(np.float64) - b.astype(np.float64))
            index = int(np.argmax(diff))
            detail = (f"max|diff|={float(diff.flatten()[index]):.3g} at flat[{index}] "
                      f"(a={float(a.flatten()[index]):.6g}, b={float(b.flatten()[index]):.6g}), "
                      f"atol={rule['atol']}")
        else:
            detail = f"allclose atol={rule['atol']} rtol={rule['rtol']}"
    return ok, detail


def sort_detections(entries: list[dict]) -> list[dict]:
    return sorted(entries, key=lambda d: (int(d["class_id"]), -float(d["score"]),
                                          float(d["x1"]), float(d["y1"]),
                                          float(d["x2"]), float(d["y2"])))


def detection_arrays(entries: list[dict]):
    boxes = np.array([[d["x1"], d["y1"], d["x2"], d["y2"]]
                      for d in entries], dtype=np.float64).reshape(-1, 4)
    scores = np.array([d["score"] for d in entries], dtype=np.float64)
    ids = np.array([d["class_id"] for d in entries], dtype=np.int64)
    return boxes, scores, ids


def check_capture_record(record: dict, role: str) -> None:
    """Validates one observer tensor record before anything is trusted."""
    name = record.get("name", "<unnamed>")
    dtype = record.get("dtype", "")
    if dtype not in KNOWN_DTYPES:
        raise ComparisonError(
            f"{role} tensor {name}: dtype {dtype!r} not in {sorted(KNOWN_DTYPES)}")
    shape = record.get("shape")
    if not isinstance(shape, list) or len(shape) != 4 or shape[0] != 1 or \
            any(int(v) <= 0 for v in shape[1:]):
        raise ComparisonError(f"{role} tensor {name}: invalid shape {shape}")
    if int(record.get("payload_bytes", 0)) <= 0 or \
            len(record.get("payload_sha256", "")) != 64 or \
            not record.get("payload_file"):
        raise ComparisonError(
            f"{role} tensor {name}: payload bytes/SHA missing — the capture must "
            "record them at write time")
    check_quant_record(record, role, name)


def check_quant_record(record: dict, role: str, name: str) -> None:
    quanti = record.get("quanti", "")
    if quanti == "scale":
        values = record.get("scale_values")
        if not isinstance(values, list) or len(values) != int(record.get("scale_len", -1)):
            raise ComparisonError(f"{role} tensor {name}: incomplete scale values")
        if any(not np.isfinite(v) for v in values):
            raise ComparisonError(f"{role} tensor {name}: non-finite scale value")
    elif quanti != "none":
        raise ComparisonError(f"{role} tensor {name}: unsupported quanti {quanti!r}")


def verify_capture_payload(directory: pathlib.Path, record: dict, role: str) -> bytes:
    path = directory / record["payload_file"]
    if not path.is_file():
        raise ComparisonError(f"{role} payload file missing: {path}")
    data = path.read_bytes()
    if len(data) != int(record["payload_bytes"]):
        raise ComparisonError(
            f"{role} payload {path.name}: size {len(data)} != recorded "
            f"{record['payload_bytes']}")
    digest = hashlib.sha256(data).hexdigest()
    if digest != record["payload_sha256"]:
        raise ComparisonError(
            f"{role} payload {path.name}: SHA-256 {digest} != recorded "
            f"{record['payload_sha256']}")
    return data


def verify_unified_payload(directory: pathlib.Path, entry: dict, role: str,
                           kind: str) -> bytes:
    """Unified payload entries MUST carry their own size and SHA-256."""
    path = directory / entry["file"]
    if not path.is_file():
        raise ComparisonError(f"{role} {kind} payload file missing: {path}")
    data = path.read_bytes()
    recorded_bytes = entry.get("bytes")
    recorded_sha = entry.get("sha256")
    if not isinstance(recorded_bytes, int) or recorded_bytes < 0:
        raise ComparisonError(
            f"{role} {kind} payload {path.name}: manifest lacks a valid bytes count")
    if len(recorded_sha or "") != 64:
        raise ComparisonError(
            f"{role} {kind} payload {path.name}: manifest lacks its SHA-256 — "
            "optional digests never pass")
    if len(data) != recorded_bytes:
        raise ComparisonError(
            f"{role} {kind} payload {path.name}: size {len(data)} != manifest "
            f"{recorded_bytes}")
    digest = hashlib.sha256(data).hexdigest()
    if digest != recorded_sha:
        raise ComparisonError(
            f"{role} {kind} payload {path.name}: SHA-256 {digest} != manifest "
            f"{recorded_sha}")
    return data


def board_conflict(target: str, identities: list[tuple[str, str]]) -> str | None:
    """Returns a failure reason when recorded board identities are not the
    comparison target, using EXACT registry aliases only.

    Each identity is (kind, reading) with kind in {"soc", "socinfo"} —
    socinfo readings resolve through the socinfo alias table (X5U -> x5).
    Every reading must resolve to exactly the comparison target; an unknown
    reading (S100Whatever) or a different concrete target (S100P under
    --target s100) is a conflict. At least one reading must exist per side.
    """
    resolved = set()
    for kind, raw in identities:
        if not raw or not raw.strip():
            continue
        reading = raw.strip().lower()
        table = TARGET_SOCINFO_ALIASES if kind == "socinfo" else TARGET_ALIASES
        match = {candidate for candidate, aliases in table.items()
                 if reading in aliases}
        if not match:
            return (f"board identity {raw!r} is not a known alias of any "
                    f"registered target — prefix matches are not identity")
        resolved.update(match)
    if not resolved:
        return "board identity unrecorded (runner and capture readings all empty)"
    if resolved != {target}:
        return (f"board identity {sorted(resolved)} conflicts with comparison "
                f"target {target!r}")
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True, choices=["x5", "s100", "s100p", "s600"])
    parser.add_argument("--repo-root", required=True, type=pathlib.Path)
    parser.add_argument("--source-capture", required=True, type=pathlib.Path)
    parser.add_argument("--unified-dump", required=True, type=pathlib.Path)
    parser.add_argument("--source-binary", required=True, type=pathlib.Path)
    parser.add_argument("--unified-binary", required=True, type=pathlib.Path)
    parser.add_argument("--unified-run-record", required=True, type=pathlib.Path,
                        help="run-record.json from run_capture.py --role unified "
                             "for the unified side's own process evidence")
    parser.add_argument("--output", required=True, type=pathlib.Path)
    args = parser.parse_args()

    output = args.output.resolve()
    if output.exists():
        print(f"refusing to overwrite existing output dir: {output}", file=sys.stderr)
        return 2
    output.mkdir(parents=True)
    failures: list[str] = []
    stages: list[dict] = []
    identity: dict = {
        "utc": utc_now(), "target": args.target, "criteria": CRITERIA,
        "detection_order_normalization":
            "both sides sorted by (class_id, -score, x1, y1, x2, y2)",
        "final_coordinate_claim":
            "final-image-coordinate comparison is REQUIRED: both sides must "
            "capture detections_original or the comparison fails",
    }
    originals = output / "originals"
    originals.mkdir()

    def bail(message: str) -> int:
        failures.append(message)
        write_report(output, identity, stages, failures, passed=False)
        print(f"comparison failed: {message}", file=sys.stderr)
        return 2

    try:
        return run_comparison(args, output, originals, identity, stages, failures, bail)
    except Exception as error:  # every failure path keeps a complete report
        traceback.print_exc(file=sys.stderr)
        return bail(f"unexpected failure: {error.__class__.__name__}: {error}")


def run_comparison(args, output, originals, identity, stages, failures, bail) -> int:
    capture_dir = args.source_capture.resolve()

    # Binary identity: hashed here from the attested files AND bound to what
    # each side recorded about its own run.
    for binary, role in ((args.source_binary, "source"), (args.unified_binary, "unified")):
        path = binary.resolve()
        if not path.is_file():
            return bail(f"{role} binary missing: {path}")
        identity[f"{role}_binary_path"] = str(path)
        identity[f"{role}_binary_sha256"] = sha256_file(path)

    # Capture state: no stale/failed/unfinished capture may ever pass.
    for marker in ("capture-in-progress.json", "capture-error.txt"):
        if (capture_dir / marker).is_file():
            return bail(f"source capture has a {marker} marker — the run did not "
                        "finish cleanly or was refused")
    capture = load_json(capture_dir / "capture.json", "source capture")
    shutil.copyfile(capture_dir / "capture.json", originals / "source-capture.json")
    if capture.get("schema") != "rdk-model-zoo/yolov5-cpp-capture/v2":
        return bail(f"unexpected capture schema: {capture.get('schema')}")
    if capture.get("failed") is True:
        return bail(f"source capture failed at runtime: {capture.get('error')!r}")
    if capture.get("return_code") != 0:
        return bail(f"source capture return_code={capture.get('return_code')}; "
                    "a failed native run never passes")
    if not capture.get("utc_start") or not capture.get("utc_finish"):
        return bail("source capture lacks start/finish UTC")

    # The external runner's process record is mandatory and cross-checked.
    run_record = load_json(capture_dir / "run-record.json", "source run record")
    shutil.copyfile(capture_dir / "run-record.json", originals / "source-run-record.json")
    if run_record.get("schema") != "rdk-model-zoo/yolov5-cpp-run-record/v1":
        return bail(f"unexpected run-record schema: {run_record.get('schema')}")
    if run_record.get("return_code") != 0:
        return bail(f"runner recorded return_code={run_record.get('return_code')} "
                    "for the source process — never a pass")
    if not run_record.get("audit_verification", {}).get("passed"):
        return bail("instrumentation audit verification failed in the run record: "
                    f"{run_record.get('audit_verification', {}).get('error')!r}")
    for role, key in (("source", "binary"), ("source", "model"), ("source", "image")):
        before = run_record.get(f"{key}_sha256_before")
        after = run_record.get(f"{key}_sha256_after")
        if not before or before != after:
            return bail(f"run record {key} hash missing or changed during the run")
    if identity["source_binary_sha256"] != run_record["binary_sha256_before"]:
        return bail("attested source binary hash != run-record hash: "
                    f"{identity['source_binary_sha256']} != "
                    f"{run_record['binary_sha256_before']}")
    for key in ("argv", "cwd", "utc_start", "utc_finish"):
        if not run_record.get(key):
            return bail(f"run record lacks {key}")
    if not (capture_dir / "stdout.txt").is_file() or \
            not (capture_dir / "stderr.txt").is_file():
        return bail("run record stdout/stderr files missing")
    shutil.copyfile(capture_dir / "stdout.txt", originals / "source-stdout.txt")
    shutil.copyfile(capture_dir / "stderr.txt", originals / "source-stderr.txt")

    unified = load_json(args.unified_dump / "manifest.json", "unified dump")
    shutil.copyfile(args.unified_dump / "manifest.json", originals / "unified-manifest.json")
    if unified.get("schema") != "rdk-model-zoo/yolov5-cpp-dump/v2":
        return bail(f"unexpected unified dump schema: {unified.get('schema')}")
    if unified.get("return_code") != 0:
        return bail(f"unified run failed (return_code={unified.get('return_code')}, "
                    f"error={unified.get('error')!r})")
    if unified.get("build_target") != args.target:
        return bail(f"unified build_target {unified.get('build_target')!r} does not "
                    f"exactly match the comparison target {args.target!r} (no "
                    "implicit s100p/s600 acceptance)")
    if not unified.get("binary_sha256"):
        return bail("unified manifest lacks its binary SHA-256")
    if unified["binary_sha256"] != identity["unified_binary_sha256"]:
        return bail(f"unified binary hash mismatch: manifest "
                    f"{unified['binary_sha256']} != attested "
                    f"{identity['unified_binary_sha256']}")
    identity["source"] = {
        "cwd": capture.get("cwd"), "argv": capture.get("argv"),
        "runner_argv": run_record.get("argv"), "runner_cwd": run_record.get("cwd"),
        "utc_start": capture.get("utc_start"), "utc_finish": capture.get("utc_finish"),
        "model_path": capture.get("model_path"), "image_path": capture.get("image_path"),
        "soc_name": capture.get("soc_name"), "board_soc": capture.get("board_soc"),
        "runner_soc_name": run_record.get("soc_name"),
        "runner_board_soc": run_record.get("board_soc"),
    }
    # The unified side needs its OWN process record: a real subprocess run
    # through the same runner, with board identity and rc/argv/binary hashes
    # bound — the manifest's build_target and internal return_code alone are
    # not physical evidence.
    unified_record = load_json(args.unified_run_record, "unified run record")
    shutil.copyfile(args.unified_run_record, originals / "unified-run-record.json")
    if unified_record.get("schema") != "rdk-model-zoo/yolov5-cpp-run-record/v1":
        return bail(f"unexpected unified run-record schema: "
                    f"{unified_record.get('schema')}")
    if unified_record.get("role") != "unified":
        return bail(f"unified run record role is {unified_record.get('role')!r}, "
                    "expected 'unified'")
    if unified_record.get("return_code") != 0:
        return bail(f"unified runner recorded return_code="
                    f"{unified_record.get('return_code')} — never a pass")
    for key in ("binary", "model", "image"):
        before = unified_record.get(f"{key}_sha256_before")
        after = unified_record.get(f"{key}_sha256_after")
        if not before or before != after:
            return bail(f"unified run record {key} hash missing or changed")
    if unified_record["binary_sha256_before"] != unified["binary_sha256"]:
        return bail("unified run-record binary hash != manifest binary_sha256")
    if unified_record["binary_sha256_before"] != identity["unified_binary_sha256"]:
        return bail("unified run-record binary hash != attested unified binary")
    if unified_record["model_sha256_before"] != capture["model_sha256"] or             unified_record["image_sha256_before"] != capture["image_sha256"]:
        return bail("unified run-record model/image hashes disagree with the "
                    "comparison identity")
    for key in ("argv", "cwd", "utc_start", "utc_finish"):
        if not unified_record.get(key):
            return bail(f"unified run record lacks {key}")
    unified_conflict = board_conflict(args.target, [
        ("socinfo", unified_record.get("soc_name", "")),
        ("soc", unified_record.get("board_soc", "")),
    ])
    if unified_conflict:
        return bail(f"unified side: {unified_conflict}")
    identity["unified"] = {
        "cwd": unified.get("cwd"), "argv": unified.get("argv"),
        "build_target": unified.get("build_target"),
        "binary_path": unified.get("binary_path"),
        "binary_sha256": unified.get("binary_sha256"),
        "runner_argv": unified_record.get("argv"),
        "runner_soc_name": unified_record.get("soc_name"),
        "runner_board_soc": unified_record.get("board_soc"),
    }
    conflict = board_conflict(args.target, [
        ("soc", capture.get("soc_name", "")),
        ("soc", capture.get("board_soc", "")),
        ("socinfo", run_record.get("soc_name", "")),
        ("soc", run_record.get("board_soc", "")),
    ])
    if conflict:
        return bail(f"source side: {conflict}")

    # Model/image identity: BOTH sides recorded execution-time hashes and they
    # must agree with each other and with the runner's pre/post hashes.
    for side, document in (("source", capture), ("unified", unified)):
        if not document.get("model_sha256") or not document.get("image_sha256"):
            return bail(f"{side} side lacks an execution-time model/image hash")
    if capture["model_sha256"] != unified["model_sha256"]:
        return bail(f"model hash mismatch: source {capture['model_sha256']} vs "
                    f"unified {unified['model_sha256']}")
    if capture["image_sha256"] != unified["image_sha256"]:
        return bail(f"image hash mismatch: source {capture['image_sha256']} vs "
                    f"unified {unified['image_sha256']}")
    if run_record["model_sha256_before"] != capture["model_sha256"] or \
            run_record["image_sha256_before"] != capture["image_sha256"]:
        return bail("run-record model/image hashes disagree with the capture-time "
                    "hashes")
    identity["model_sha256"] = capture["model_sha256"]
    identity["image_sha256"] = capture["image_sha256"]

    # Thresholds: the real v2 manifest stores parameters as a JSON object, and
    # the comparison is float32-bit exact (native semantics, no widened
    # tolerance; "0.450000" and 0.44999998807907104 are the same float32).
    parameters = unified.get("parameters")
    if not isinstance(parameters, dict):
        return bail(f"unified parameters must be a JSON object (v2 schema), got "
                    f"{type(parameters).__name__}")
    for threshold, capture_key, unified_key in (
            ("score", "score_threshold", "score_thres"),
            ("nms", "nms_threshold", "nms_thres")):
        if unified_key not in parameters:
            return bail(f"unified dump lacks parameter {unified_key}")
        if not f32_equal(float(capture[capture_key]), float(parameters[unified_key])):
            return bail(f"{threshold} threshold float32-bit mismatch: source "
                        f"{capture[capture_key]!r} vs unified "
                        f"{parameters[unified_key]!r}")

    # Structural counts: an empty or short list can never bypass comparison.
    capture_inputs = capture.get("inputs", [])
    unified_inputs = unified.get("input_tensors", [])
    if len(capture_inputs) != EXPECTED_INPUTS[args.target]:
        return bail(f"source capture has {len(capture_inputs)} inputs; target "
                    f"{args.target} requires exactly {EXPECTED_INPUTS[args.target]}")
    if len(unified_inputs) != EXPECTED_INPUTS[args.target]:
        return bail(f"unified dump has {len(unified_inputs)} input payloads; target "
                    f"{args.target} requires exactly {EXPECTED_INPUTS[args.target]}")
    capture_outputs = capture.get("outputs", [])
    unified_outputs_list = unified.get("outputs", [])
    unified_raw = unified.get("raw_tensors", [])
    if len(capture_outputs) != EXPECTED_OUTPUTS:
        return bail(f"source capture has {len(capture_outputs)} outputs; exactly "
                    f"{EXPECTED_OUTPUTS} heads are required")
    if len(unified_outputs_list) != EXPECTED_OUTPUTS or len(unified_raw) != EXPECTED_OUTPUTS:
        return bail(f"unified dump has {len(unified_outputs_list)} output infos / "
                    f"{len(unified_raw)} raw payloads; exactly {EXPECTED_OUTPUTS} "
                    "heads are required")
    for record in capture_inputs + capture_outputs:
        check_capture_record(record, "source")
    # Unified tensor info is validated with the same teeth: dtype, shape,
    # quantization descriptor — and payload entries must carry bytes+SHA.
    for entry in unified_inputs + unified_raw:
        dtype = entry.get("dtype", "")
        if dtype not in KNOWN_DTYPES:
            return bail(f"unified payload {entry.get('name')}: dtype {dtype!r} not "
                        f"in {sorted(KNOWN_DTYPES)}")
    unified_output_names = [entry["name"] for entry in unified_outputs_list]
    if len(set(unified_output_names)) != EXPECTED_OUTPUTS:
        return bail("unified output names are not unique")
    for entry in unified_outputs_list:
        shape = entry.get("shape")
        if not isinstance(shape, list) or len(shape) != 4 or shape[0] != 1 or \
                any(int(v) <= 0 for v in shape[1:]):
            return bail(f"unified output {entry.get('name')}: invalid shape {shape}")
        check_quant_record(entry, "unified", entry["name"])

    # Stage 1: inputs (exact after stride restoration; positional match, with
    # dtype and shape agreement enforced on both sides).
    for record, unified_record in zip(capture_inputs, unified_inputs):
        name = record["name"]
        if unified_record.get("dtype") != record["dtype"]:
            return bail(f"input {name}: dtype {record['dtype']} vs unified "
                        f"{unified_record.get('dtype')}")
        if [int(v) for v in unified_record.get("shape", [])] != \
                [int(v) for v in record["shape"]]:
            return bail(f"input {name}: shape {record['shape']} vs unified "
                        f"{unified_record.get('shape')}")
        source_bytes = verify_capture_payload(capture_dir, record, "source")
        unified_bytes = verify_unified_payload(args.unified_dump, unified_record,
                                               "unified", "input")
        if args.target == "x5":
            source_logical = source_bytes
            unified_logical = unified_bytes
            note = "compact NV12 payload compared byte-exact"
        else:
            source_logical = restore_rows(source_bytes, record["shape"], record["stride"])
            unified_logical = unified_bytes  # row-gathered by the dump writer
            note = "stride-restored plane rows compared byte-exact"
        np.save(output / f"input-{name}-source.npy",
                np.frombuffer(source_logical, dtype=np.uint8))
        np.save(output / f"input-{name}-unified.npy",
                np.frombuffer(unified_logical, dtype=np.uint8))
        shutil.copyfile(capture_dir / record["payload_file"],
                        originals / f"input-{name}-source-physical.bin")
        shutil.copyfile(args.unified_dump / unified_record["file"],
                        originals / f"input-{name}-unified-physical.bin")
        ok = source_logical == unified_logical
        stages.append(stage_report(f"input:{name}", ok,
                                   note if ok else "logical input bytes differ"))
        if not ok:
            failures.append(f"input {name} differs")

    # Stage 2: raw outputs, matched by a complete shape bijection, with
    # quantization descriptors enforced on BOTH sides.
    unified_by_shape: dict[tuple, dict] = {}
    for entry in unified_outputs_list:
        shape = tuple(int(v) for v in entry["shape"])
        if shape in unified_by_shape:
            return bail(f"duplicate unified output shape {shape} — matching is "
                        "ambiguous")
        unified_by_shape[shape] = entry
    capture_shapes = [tuple(int(v) for v in record["shape"]) for record in capture_outputs]
    if len(set(capture_shapes)) != EXPECTED_OUTPUTS:
        return bail(f"source output shapes are not unique: {capture_shapes}")
    for shape in capture_shapes:
        if shape not in unified_by_shape:
            return bail(f"no unified output with shape {shape} — missing head")
    if len(unified_by_shape) != EXPECTED_OUTPUTS:
        return bail("unified has extra output shapes beyond the source heads")

    for record, shape in zip(capture_outputs, capture_shapes):
        name = record["name"]
        partner = unified_by_shape[shape]
        partner_name = partner["name"]
        if partner.get("dtype") != record["dtype"]:
            return bail(f"raw head {shape}: dtype {record['dtype']} vs unified "
                        f"{partner.get('dtype')} — never masked by a float cast")
        if partner.get("quanti") != record["quanti"]:
            return bail(f"raw head {shape}: quantization kind {record['quanti']!r} "
                        f"vs unified {partner.get('quanti')!r}")
        if record["quanti"] == "scale":
            if int(partner.get("scale_len", -1)) != int(record["scale_len"]):
                return bail(f"scale descriptor length of {name}: "
                            f"{record['scale_len']} vs unified "
                            f"{partner.get('scale_len')}")
            if int(record.get("quantize_axis", -1)) != int(partner.get("quantize_axis", -2)):
                return bail(f"quantize axis of {name}: {record.get('quantize_axis')} "
                            f"vs unified {partner.get('quantize_axis')}")
        source_bytes = verify_capture_payload(capture_dir, record, "source")
        unified_entry = next(entry for entry in unified_raw
                             if entry["name"] == partner_name)
        unified_bytes = verify_unified_payload(args.unified_dump, unified_entry,
                                               "unified", "raw")
        source_logical = restore_logical(source_bytes, record["dtype"], list(shape),
                                         record["stride"])
        unified_logical = restore_logical(
            unified_bytes, partner["dtype"], partner["shape"],
            sanitize_stride(partner.get("stride")))
        np.save(output / f"raw-{name}-source.npy", source_logical)
        np.save(output / f"raw-{partner_name}-unified.npy", unified_logical)
        shutil.copyfile(capture_dir / record["payload_file"],
                        originals / f"raw-{name}-source-physical.bin")
        shutil.copyfile(args.unified_dump / unified_entry["file"],
                        originals / f"raw-{partner_name}-unified-physical.bin")
        ok, detail = compare_arrays(source_logical, unified_logical, CRITERIA["raw"],
                                    require_same_dtype=True)
        stages.append(stage_report(f"raw:{name}->{partner_name}", ok, detail))
        if not ok:
            failures.append(f"raw output {name} (shape {shape}) differs: {detail}")
        if record["quanti"] == "scale":
            source_scale = np.asarray(record["scale_values"], dtype=np.float32)
            unified_scale = np.asarray(partner.get("scale_values", []), dtype=np.float32)
            scale_ok = (source_scale.shape == unified_scale.shape and
                        source_scale.tobytes() == unified_scale.tobytes())
            source_zero = record.get("zero_point_values")
            unified_zero = partner.get("zero_point_values")
            zero_ok = source_zero == unified_zero
            for label, ok_value in (("scale", scale_ok), ("zero", zero_ok)):
                stages.append(stage_report(
                    f"{label}:{name}", ok_value,
                    "float32-bit exact" if ok_value else f"{label} differs"))
                if not ok_value:
                    failures.append(f"{label} descriptor of {name} differs")

    # Stage 3: detections. The keys themselves are required — a missing key
    # never masquerades as an observed empty list — and final-image
    # coordinates are required on BOTH sides.
    for side, document, key in (("source", capture, "detections"),
                                ("source", capture, "detections_original"),
                                ("unified", unified, "detections"),
                                ("unified", unified, "detections_original")):
        if key not in document:
            return bail(f"{side} manifest lacks the {key} key — absent evidence "
                        "never passes as an observed empty result")

    def detection_stage(label: str, source_list: list[dict], unified_list: list[dict]) -> None:
        np.save(output / f"detections-{label}-source.npy", np.array(
            [[d["x1"], d["y1"], d["x2"], d["y2"], d["score"], d["class_id"]]
             for d in source_list], dtype=np.float64).reshape(-1, 6))
        np.save(output / f"detections-{label}-unified.npy", np.array(
            [[d["x1"], d["y1"], d["x2"], d["y2"], d["score"], d["class_id"]]
             for d in unified_list], dtype=np.float64).reshape(-1, 6))
        if len(source_list) != len(unified_list):
            failures.append(f"{label} detection counts differ: source "
                            f"{len(source_list)} vs unified {len(unified_list)}; both "
                            "full arrays are saved")
            stages.append(stage_report(f"{label}:count", False,
                                       f"{len(source_list)} vs {len(unified_list)}"))
            return
        if not source_list:
            stages.append(stage_report(f"{label}:empty", True, "both sides empty"))
            return
        s_boxes, s_scores, s_ids = detection_arrays(source_list)
        u_boxes, u_scores, u_ids = detection_arrays(unified_list)
        for sub, a, b, rule in (
                ("boxes", s_boxes, u_boxes, CRITERIA["boxes"]),
                ("scores", s_scores, u_scores, CRITERIA["scores"]),
                ("class_ids", s_ids, u_ids, CRITERIA["class_ids"])):
            ok, detail = compare_arrays(a, b, rule)
            stages.append(stage_report(f"{label}:{sub}", ok, detail))
            if not ok:
                failures.append(f"{label} {sub} differ: {detail}")

    detection_stage("model-space", sort_detections(capture["detections"]),
                    sort_detections(unified["detections"]))
    detection_stage("final-coordinates",
                    sort_detections(capture["detections_original"]),
                    sort_detections(unified["detections_original"]))

    digests = {}
    for path in sorted(output.rglob("*")):
        if path.is_file():
            digests[str(path.relative_to(output))] = sha256_file(path)
    (output / "digests.json").write_text(
        json.dumps(digests, indent=2) + "\n", encoding="utf-8")

    passed = not failures
    write_report(output, identity, stages, failures, passed)
    print(json.dumps({"passed": passed, "output": str(output),
                      "failures": failures}, indent=2))
    return 0 if passed else 2


def write_report(output: pathlib.Path, identity: dict, stages: list[dict],
                 failures: list[str], passed: bool) -> None:
    report = {
        "schema": "rdk-model-zoo/yolov5-cpp-comparison/v2",
        "utc": utc_now(),
        "passed": passed,
        "failures": failures,
        "identity": identity,
        "stages": stages,
    }
    (output / "comparison.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
