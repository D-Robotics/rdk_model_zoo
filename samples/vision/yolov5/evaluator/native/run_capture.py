#!/usr/bin/env python3
# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Run the instrumented fixed-source binary and record REAL process evidence.

The C++ observer (ycap_observer.hpp) only records what cannot be observed from
outside the process (tensor payloads, stage metadata). Everything process-
level — the true subprocess argv (including arguments gflags strips from the
program's own argc/argv), separate stdout and stderr, the real exit code,
start/end UTC, the working directory, board identity and the pre/post hashes
of the binary, model and image — is captured HERE, around the actual run, and
written to <capture-dir>/run-record.json. The comparison tool requires this
record and cross-checks it, so a capture can never masquerade as a process it
was not.

The capture directory must be empty (same refusal rule as the observer).
The instrumentation audit is verified against the actual generated files, so
the run is bound to the exact instrumented code it executed.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import pathlib
import subprocess
import sys


def utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 16), b""):
            digest.update(block)
    return digest.hexdigest()


def read_first_line(path: str) -> str:
    try:
        with open(path) as handle:
            return handle.readline().strip()
    except OSError:
        return ""


def verify_audit(audit_path: pathlib.Path, work_dir: pathlib.Path,
                 audit_bytes: bytes | None = None) -> dict:
    """Verifies the instrumentation audit against the protocol instrument.py
    actually emits and against the generated work dir.

    Required: schema, target with its pinned commit, the exact pinned source
    set with every anchor matched exactly once, blob_sha256 equal to the
    pinned SHA, the complete per-target closure, the observer header and
    CMake hashes, and every recorded file present with its recorded hash.
    Empty or truncated audits are rejected outright."""
    try:
        audit = json.loads(audit_bytes if audit_bytes is not None else audit_path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        return {"passed": False, "error": f"unreadable audit: {error}"}
    if not isinstance(audit, dict) or not audit:
        return {"passed": False, "error": "audit is empty or not an object"}
    if audit.get("schema") != "rdk-model-zoo/yolov5-cpp-instrumentation/v1":
        return {"passed": False,
                "error": f"unknown audit schema: {audit.get('schema')!r}"}
    target = audit.get("target")
    if target not in ("x5", "s100"):
        return {"passed": False, "error": f"unknown audit target: {target!r}"}
    import importlib.util
    instrument_path = pathlib.Path(__file__).resolve().parent / "instrument.py"
    spec = importlib.util.spec_from_file_location("b7_instrument_check",
                                                  instrument_path)
    instrument = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(instrument)
    pinned_commit = {"x5": instrument.X5_COMMIT, "s100": instrument.S_COMMIT}[target]
    if audit.get("pinned_commit") != pinned_commit:
        return {"passed": False,
                "error": f"audit pinned_commit {audit.get('pinned_commit')!r} != "
                         f"expected {pinned_commit}"}
    expected_sources = dict(instrument.PINNED_SHA256[target])
    expected_closure = dict(instrument.S_CLOSURE_SHA256) if target != "x5" else {}
    sources = audit.get("sources")
    if not isinstance(sources, list) or not sources:
        return {"passed": False, "error": "audit has no sources"}
    checked = []
    seen_sources = set()
    for source in sources:
        if not isinstance(source, dict):
            return {"passed": False, "error": "malformed audit source entry"}
        path = source.get("source_path")
        if path not in expected_sources:
            return {"passed": False,
                    "error": f"unexpected source {path!r} (expected exactly "
                             f"{sorted(expected_sources)})"}
        if path in seen_sources:
            return {"passed": False, "error": f"duplicate source {path!r}"}
        seen_sources.add(path)
        if source.get("blob_sha256") != expected_sources[path]:
            return {"passed": False,
                    "error": f"source {path}: blob SHA {source.get('blob_sha256')!r} "
                             f"!= pinned {expected_sources[path]}"}
        if source.get("pinned_sha256") != expected_sources[path]:
            return {"passed": False, "error": f"source {path}: pinned SHA mismatch"}
        anchors = source.get("anchors")
        if not isinstance(anchors, list) or not anchors or \
                any(entry.get("matched") != 1 for entry in anchors):
            return {"passed": False,
                    "error": f"source {path}: anchors not all matched exactly once"}
        instrumented = work_dir / path.replace(
            "samples/vision/yolov5/runtime/cpp/", "")
        if not instrumented.is_file():
            return {"passed": False,
                    "error": f"instrumented source missing on disk: {instrumented}"}
        observed = sha256_file(instrumented)
        if observed != source.get("instrumented_sha256"):
            return {"passed": False,
                    "error": f"instrumented mismatch for {instrumented}: "
                             f"{observed} != {source.get('instrumented_sha256')}"}
        checked.append(str(instrumented))
    if seen_sources != set(expected_sources):
        return {"passed": False,
                "error": f"audit sources {sorted(seen_sources)} != expected "
                         f"{sorted(expected_sources)}"}
    closure = audit.get("closure", [])
    if target != "x5":
        if not isinstance(closure, list) or not closure:
            return {"passed": False,
                    "error": f"target {target} requires a non-empty closure"}
        seen_closure = set()
        for item in closure:
            path = item.get("source_path")
            if path not in expected_closure:
                return {"passed": False,
                        "error": f"unexpected closure entry {path!r}"}
            if path in seen_closure:
                return {"passed": False, "error": f"duplicate closure {path!r}"}
            seen_closure.add(path)
            if item.get("blob_sha256") != expected_closure[path]:
                return {"passed": False,
                        "error": f"closure {path}: blob SHA != pinned "
                                 f"{expected_closure[path]}"}
            copy = work_dir / item.get("work_copy", "")
            if not copy.is_file():
                return {"passed": False,
                        "error": f"closure copy missing: {copy}"}
            observed = sha256_file(copy)
            if observed != item.get("blob_sha256"):
                return {"passed": False,
                        "error": f"closure mismatch for {copy}: {observed} != "
                                 f"{item.get('blob_sha256')}"}
            checked.append(str(copy))
        if seen_closure != set(expected_closure):
            return {"passed": False,
                    "error": f"audit closure {sorted(seen_closure)} != expected "
                             f"{sorted(expected_closure)}"}
    elif closure:
        return {"passed": False, "error": "x5 audit must not carry a closure"}
    # The observer header and the generated CMake are part of the build.
    header = work_dir / "ycap_observer.hpp"
    if not header.is_file() or sha256_file(header) != audit.get("hooks_sha256"):
        return {"passed": False,
                "error": "observer header missing or hash mismatch"}
    checked.append(str(header))
    cmake = work_dir / "CMakeLists.txt"
    if not cmake.is_file() or sha256_file(cmake) != audit.get("cmake_sha256"):
        return {"passed": False,
                "error": "generated CMakeLists missing or hash mismatch"}
    checked.append(str(cmake))
    return {"passed": True, "verified_files": checked}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True, type=pathlib.Path)
    parser.add_argument("--capture-dir", required=True, type=pathlib.Path)
    parser.add_argument("--model", required=True, type=pathlib.Path)
    parser.add_argument("--image", required=True, type=pathlib.Path)
    parser.add_argument("--role", default="source", choices=["source", "unified"],
                        help="source: the instrumented fixed binary (audit "
                             "required); unified: the unified yolov5_cpp "
                             "(no instrumentation audit, still full process "
                             "evidence)")
    parser.add_argument("--audit", required=False, type=pathlib.Path,
                        help="instrumentation-audit.json of the generated build "
                             "(required for --role source)")
    parser.add_argument("--cwd", default=".", help="working directory for the run")
    parser.add_argument("command_args", nargs=argparse.REMAINDER,
                        help="arguments after '--' are passed to the binary")
    args = parser.parse_args()

    binary = args.binary.resolve()
    capture_dir = args.capture_dir.resolve()
    model = args.model.resolve()
    image = args.image.resolve()
    if args.role == "source" and args.audit is None:
        print("run_capture: --audit is required for --role source", file=sys.stderr)
        return 2
    audit_path = args.audit.resolve() if args.audit is not None else None
    work_dir = audit_path.parent.resolve() if audit_path is not None else capture_dir
    for label, path in (("binary", binary), ("model", model), ("image", image)):
        if not path.is_file():
            print(f"run_capture: {label} missing: {path}", file=sys.stderr)
            return 2
    if audit_path is not None and not audit_path.is_file():
        print(f"run_capture: audit missing: {audit_path}", file=sys.stderr)
        return 2
    if capture_dir.exists() and any(capture_dir.iterdir()):
        print(f"run_capture: refusing non-empty capture dir: {capture_dir}",
              file=sys.stderr)
        return 2
    capture_dir.mkdir(parents=True, exist_ok=True)

    pass_through = list(args.command_args)
    if pass_through and pass_through[0] == "--":
        pass_through = pass_through[1:]
    binary_before = sha256_file(binary)
    model_before = sha256_file(model)
    image_before = sha256_file(image)
    # Hold the exact bytes verified before execution. Persist them AFTER the
    # child exits, because the C++ observer requires an empty capture directory.
    try:
        audit_bytes = audit_path.read_bytes() if args.role == "source" else None
    except OSError as error:
        print(f"run_capture: unreadable audit: {error}", file=sys.stderr)
        return 2
    audit_binding = ({"audit_file": "instrumentation-audit.json",
                      "audit_sha256": hashlib.sha256(audit_bytes).hexdigest()}
                     if audit_bytes is not None else {})
    audit_check = (verify_audit(audit_path, work_dir, audit_bytes)
                   if args.role == "source"
                   else {"passed": True, "verified_files": [],
                         "note": "unified role: no instrumentation audit"})
    if not audit_check["passed"]:
        # The binary is NEVER run when the audit fails: a stale or tampered
        # instrumented build must not produce evidence. The failure record is
        # still persisted with the audit verdict.
        record = {
            "schema": "rdk-model-zoo/yolov5-cpp-run-record/v1",
            "role": args.role,
            "utc_start": utc_now(), "utc_finish": utc_now(),
            "binary_path": str(binary),
            "binary_sha256_before": sha256_file(binary),
            "model_path": str(model), "model_sha256_before": sha256_file(model),
            "image_path": str(image), "image_sha256_before": sha256_file(image),
            "audit_path": str(audit_path), "audit_verification": audit_check,
            **audit_binding,
            "cwd": str(pathlib.Path(args.cwd).resolve()),
            "soc_name": read_first_line("/sys/class/socinfo/soc_name"),
            "board_soc": read_first_line("/sys/class/boardinfo/soc_name"),
            "argv": [str(binary)] + (list(args.command_args)[1:]
                                     if list(args.command_args)[:1] == ["--"]
                                     else list(args.command_args)),
            "return_code": None,
            "binary_sha256_after": sha256_file(binary),
            "model_sha256_after": sha256_file(model),
            "image_sha256_after": sha256_file(image),
            "error": "audit verification failed; binary not executed",
        }
        if audit_bytes is not None:
            (capture_dir / "instrumentation-audit.json").write_bytes(audit_bytes)
        (capture_dir / "stdout.txt").write_text("", encoding="utf-8")
        (capture_dir / "stderr.txt").write_text(
            f"run_capture: {record['error']}\n", encoding="utf-8")
        (capture_dir / "run-record.json").write_text(
            json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"run_capture: audit verification failed, binary not executed: "
              f"{audit_check['error']}", file=sys.stderr)
        return 2
    record = {
        "schema": "rdk-model-zoo/yolov5-cpp-run-record/v1",
        "role": args.role,
        "utc_start": utc_now(),
        "binary_path": str(binary),
        "binary_sha256_before": binary_before,
        "model_path": str(model),
        "model_sha256_before": model_before,
        "image_path": str(image),
        "image_sha256_before": image_before,
        "audit_path": str(audit_path),
        "audit_verification": audit_check,
        **audit_binding,
        "cwd": str(pathlib.Path(args.cwd).resolve()),
        "soc_name": read_first_line("/sys/class/socinfo/soc_name"),
        "board_soc": read_first_line("/sys/class/boardinfo/soc_name"),
        "argv": [str(binary)] + pass_through,
    }

    environment = dict(os.environ)
    environment["YOLOV5_CAPTURE_DIR"] = str(capture_dir)
    try:
        completed = subprocess.run(
            record["argv"], cwd=record["cwd"], env=environment,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=3600)
        record["return_code"] = completed.returncode
        record["stdout"] = completed.stdout.decode(errors="replace")
        record["stderr"] = completed.stderr.decode(errors="replace")
    except subprocess.TimeoutExpired as error:
        record["return_code"] = -1
        record["stdout"] = (error.stdout or b"").decode(errors="replace")
        record["stderr"] = "run_capture: timeout\n" + (error.stderr or b"").decode(
            errors="replace")
    except OSError as error:
        record["return_code"] = -1
        record["stdout"] = ""
        record["stderr"] = f"run_capture: {error}\n"

    record["utc_finish"] = utc_now()
    # Post-run hashes bind the record to bytes that still exist unchanged.
    record["binary_sha256_after"] = sha256_file(binary) if binary.is_file() else ""
    record["model_sha256_after"] = sha256_file(model) if model.is_file() else ""
    record["image_sha256_after"] = sha256_file(image) if image.is_file() else ""

    if audit_bytes is not None:
        (capture_dir / "instrumentation-audit.json").write_bytes(audit_bytes)
    (capture_dir / "stdout.txt").write_text(record.pop("stdout"), encoding="utf-8")
    (capture_dir / "stderr.txt").write_text(record.pop("stderr"), encoding="utf-8")
    (capture_dir / "run-record.json").write_text(
        json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"capture_dir": str(capture_dir),
                      "return_code": record["return_code"],
                      "audit_verified": audit_check["passed"]}, indent=2))
    return 0 if record["return_code"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
