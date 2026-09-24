# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Host behaviour tests for the fixed-source C++ comparison tooling.

Covers samples/vision/yolov5/evaluator/native/: instrument.py (SHA pinning
incl. the build closure, unique anchors, fail-closed generation), the observer
header (compiles standalone, SDK-free, real generated hook ORDER), the
run_capture.py process-evidence runner, and compare_native.py (stride-restored
comparison against the REAL v2 board-manifest schema with every bypass the
independent reviews found closed). Compiling the instrumented sources against
host stubs only proves the injected glue compiles — it is NOT evidence that a
real SDK build passed, which stays a board concern.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import pathlib
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
NATIVE = ROOT / "samples" / "vision" / "yolov5" / "evaluator" / "native"
TEST_CPP = ROOT / "samples" / "vision" / "yolov5" / "tests" / "test_cpp_contract.py"
# Verbatim copy of the coordinator's on-board unified dump manifest (4d45f9a,
# X5 8GB); kept under tests/data/native so this regression runs in EVERY
# checkout. See tests/data/native/PROVENANCE.md.
REAL_MANIFEST = ROOT / "samples" / "vision" / "yolov5" / "tests" / "data" / \
    "native" / "x5-board-manifest.json"

PY = sys.executable
F32 = lambda v: struct.unpack("f", struct.pack("f", v))[0]  # noqa: E731


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_file(path: Path, data: bytes) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return {"file": str(path.name), "bytes": len(data), "sha256": sha(data)}


def make_run_record(capture_dir: Path, *, model: bytes, image: bytes,
                    source_binary: bytes, rc: int = 0, soc: str = "S100",
                    audit_passed: bool = True, binary_changed: bool = False) -> None:
    (capture_dir / "stdout.txt").write_text("ran\n")
    (capture_dir / "stderr.txt").write_text("")
    record = {
        "schema": "rdk-model-zoo/yolov5-cpp-run-record/v1",
        "utc_start": "2026-09-24T00:00:00Z", "utc_finish": "2026-09-24T00:00:03Z",
        "binary_path": "/tmp/yolov5_fixed_capture",
        "binary_sha256_before": sha(source_binary),
        "binary_sha256_after": sha(source_binary + (b"x" if binary_changed else b"")),
        "model_path": "/tmp/m.hbm", "model_sha256_before": sha(model),
        "model_sha256_after": sha(model),
        "image_path": "/tmp/i.jpg", "image_sha256_before": sha(image),
        "image_sha256_after": sha(image),
        "audit_path": "/tmp/audit.json",
        "audit_verification": {"passed": audit_passed},
        "cwd": "/tmp", "soc_name": soc, "board_soc": soc,
        "argv": ["/tmp/yolov5_fixed_capture", "--flag", "kept-by-runner"],
        "return_code": rc,
    }
    (capture_dir / "run-record.json").write_text(json.dumps(record), encoding="utf-8")


def build_s100_capture(directory: Path, *, tamper_raw: bool = False,
                       fail_marker: str = "", soc: str = "S100",
                       omit_run_record: bool = False,
                       drop_detections_key: bool = False,
                       drop_original_key: bool = False) -> Path:
    """A structurally real S capture: v2 schema, run record, two NV12 planes,
    three heads with distinct shapes and padded strides, float32-exact
    threshold/scale values that are NOT binary-representable decimals."""
    directory.mkdir(parents=True)
    model, image = b"model-bytes", b"image-bytes"
    (directory / "model.hbm").write_bytes(model)
    (directory / "image.jpg").write_bytes(image)
    if fail_marker:
        (directory / fail_marker).write_text("stale\n")
    if not omit_run_record:
        make_run_record(directory, model=model, image=image,
                        source_binary=b"source-binary", soc=soc)
    plane0 = bytes([1, 2, 3, 4, 0xAA, 0xAA, 0xAA, 0xAA,
                    5, 6, 7, 8, 0xAA, 0xAA, 0xAA, 0xAA])
    plane1 = bytes([9, 10, 11, 12, 13, 14, 15, 16])
    payloads = {"input0": plane0, "input1": plane1}

    def head(values, padding):
        raw = bytearray()
        for pixel in values:
            for value in pixel:
                raw += struct.pack("<i", value)
            raw += b"\xBB" * padding
        return bytes(raw)

    head0 = head([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], 4)
    if tamper_raw:
        head0 = struct.pack("<i", 999) + head0[4:]
    head1 = head([[1, 2, 3, 4], [5, 6, 7, 8]], 0)
    head2 = head([[21, 22, 23]], 0)
    payloads.update({"output0": head0, "output1": head1, "output2": head2})

    # float32 values whose decimal spellings differ between 9- and 17-digit
    # serializers but whose float32 bits are identical.
    scale0 = F32(0.00418911874294281)
    scale1 = F32(0.45)

    def tensor(name, dtype, shape, stride, aligned, quanti, scale_len, values,
               payload, inputs):
        written = write_file(directory / f"payload-{name}.bin", payloads[name])
        return {
            "name": name, "dtype": dtype, "shape": shape, "stride": stride,
            "aligned_byte_size": aligned, "quanti": quanti, "scale_len": scale_len,
            "zero_point_len": 0, "quantize_axis": 3 if quanti == "scale" else -1,
            "scale_values": values, "zero_point_values": [],
            "payload_file": written["file"], "payload_bytes": written["bytes"],
            "payload_sha256": written["sha256"],
        }

    capture = {
        "schema": "rdk-model-zoo/yolov5-cpp-capture/v2",
        "utc_start": "2026-09-24T00:00:00Z", "utc_finish": "2026-09-24T00:00:02Z",
        "cwd": "/tmp", "argv": "yolov5_fixed_capture --flag",
        "model_path": str(directory / "model.hbm"), "model_sha256": sha(model),
        "image_path": str(directory / "image.jpg"), "image_sha256": sha(image),
        "label_path": "", "score_threshold": F32(0.25), "nms_threshold": F32(0.45),
        "soc_name": soc, "board_soc": soc, "return_code": 0, "failed": False,
        "error": "", "warning": "",
        "inputs": [
            tensor("input0", "uint8", [1, 2, 4, 1], [16, 8, 1, 1], 16, "none", 0, [], "input0", True),
            tensor("input1", "uint8", [1, 1, 4, 2], [8, 8, 2, 1], 8, "none", 0, [], "input1", True),
        ],
        "outputs": [
            tensor("output0", "int32", [1, 2, 2, 3], [64, 32, 16, 4], 64, "scale", 3, [scale0, 0.25, 0.125], "output0", False),
            tensor("output1", "int32", [1, 2, 1, 4], [32, 16, 16, 4], 32, "scale", 4, [scale1, 0.25, 0.125, 0.0625], "output1", False),
            tensor("output2", "int32", [1, 1, 1, 3], [12, 12, 12, 4], 12, "none", 0, [], "output2", False),
        ],
        "detections": [
            {"x1": 8.0, "y1": 9.0, "x2": 40.0, "y2": 41.0, "score": 0.75, "class_id": 5},
            {"x1": 1.0, "y1": 2.0, "x2": 3.0, "y2": 4.0, "score": 0.5, "class_id": 0}],
        "detections_original": [
            {"x1": 16.0, "y1": 18.0, "x2": 80.0, "y2": 82.0, "score": 0.75, "class_id": 5},
            {"x1": 2.0, "y1": 4.0, "x2": 6.0, "y2": 8.0, "score": 0.5, "class_id": 0}],
    }
    if drop_detections_key:
        capture.pop("detections")
    if drop_original_key:
        capture.pop("detections_original")
    (directory / "capture.json").write_text(json.dumps(capture), encoding="utf-8")
    return directory


def build_unified_dump(directory: Path, *, rc: int = 0, wrong_model: bool = False,
                       omit_raw: bool = False, parameters_style: str = "dict",
                       output_count: int = 3, dtype_override: str = None,
                       empty_lists: bool = False, extra_output: bool = False,
                       omit_hashes: bool = False, include_original_dets: bool = True,
                       build_target: str = "s100", drop_payload_hashes: bool = False,
                       flip_quanti: bool = False,
                       nine_digit_scales: bool = True) -> Path:
    directory.mkdir(parents=True)
    model, image = b"model-bytes", b"image-bytes"
    unified_binary = b"unified-binary"

    def write_relative(relative: str, data: bytes) -> dict:
        written = write_file(directory / relative, data)
        entry = {**written, "file": relative}
        if drop_payload_hashes:
            entry.pop("sha256")
            entry["bytes"] = -123
        return entry

    in0 = write_relative("input/0-input0-y.bin", bytes([1, 2, 3, 4, 5, 6, 7, 8]))
    in1 = write_relative("input/1-input1-uv.bin", bytes([9, 10, 11, 12, 13, 14, 15, 16]))

    def head(values, padding):
        raw = bytearray()
        for pixel in values:
            for value in pixel:
                raw += struct.pack("<i", value)
            raw += b"\xCC" * padding
        return bytes(raw)

    scale0 = F32(0.00418911874294281)
    scale1 = F32(0.45)

    def spelling(value: float) -> float:
        # The unified C++ dump serializes floats with 9 significant digits.
        return float(format(value, ".9g")) if nine_digit_scales else value

    raws = [
        ("output0", head([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], 0), 48,
         [1, 2, 2, 3], "int32", "scale", 3, [scale0, 0.25, 0.125], [0, 0, 0, 0]),
        ("output1", head([[1, 2, 3, 4], [5, 6, 7, 8]], 0), 32,
         [1, 2, 1, 4], "int32", "scale", 4, [scale1, 0.25, 0.125, 0.0625], [0, 0, 0, 0]),
        ("output2", head([[21, 22, 23]], 0), 12,
         [1, 1, 1, 3], "int32", "none", 0, [], [0, 0, 0, 0]),
    ]
    raws = raws[:output_count]
    if extra_output:
        raws.append(("output3", head([[3, 2, 1]], 0), 12, [1, 1, 1, 4], "int32",
                     "none", 0, [], [0, 0, 0, 0]))
    raw_entries = []
    output_infos = []
    for index, (name, data, size, shape, dtype, quanti, scale_len, scales, stride) in \
            enumerate(raws):
        written = write_relative(f"raw/{index}-{name}.bin", data)
        raw_entries.append({"name": name, "dtype": dtype, "shape": shape[1:],
                            **written})
        effective_quanti = ("none" if quanti == "scale" else "scale") if flip_quanti \
            else quanti
        output_infos.append({
            "name": name, "dtype": dtype_override or dtype, "shape": shape,
            "quanti": effective_quanti, "scale_len": scale_len,
            "aligned_byte_size": size, "stride": stride,
            "aligned": [None, None, None, None],
            "quantize_axis": 3 if quanti == "scale" else 0,
            "scale_values": [spelling(v) for v in scales],
            "zero_point_values": []})
    if empty_lists:
        raw_entries, output_infos = [], []
    parameters = {"score_thres": "0.250000", "nms_thres": "0.450000"}
    if parameters_style == "pairs":
        parameters = [["score_thres", "0.250000"], ["nms_thres", "0.450000"]]
    manifest = {
        "schema": "rdk-model-zoo/yolov5-cpp-dump/v2",
        "utc": "2026-09-24T00:00:01Z", "target": build_target,
        "build_target": build_target, "asset_id": "s:yolov5:x-672",
        "model_path": "/tmp/model.hbm",
        "model_sha256": "0" * 64 if wrong_model else sha(model),
        "binary_path": "/tmp/yolov5_cpp", "binary_sha256": sha(unified_binary),
        "image_path": "/tmp/image.jpg", "image_sha256": sha(image),
        "cwd": "/tmp", "argv": ["yolov5_cpp"], "return_code": rc, "error": "",
        "parameters": parameters,
        "input_tensors": [
            {"name": "input0-y", "dtype": "uint8", "shape": [1, 2, 4, 1], **in0},
            {"name": "input1-uv", "dtype": "uint8", "shape": [1, 1, 4, 2], **in1}],
        "outputs": output_infos,
        "detections": [
            {"x1": 1.0, "y1": 2.0, "x2": 3.0, "y2": 4.0, "score": 0.5, "class_id": 0},
            {"x1": 8.0, "y1": 9.0, "x2": 40.0, "y2": 41.0, "score": 0.75, "class_id": 5}],
    }
    if include_original_dets:
        manifest["detections_original"] = [
            {"x1": 2.0, "y1": 4.0, "x2": 6.0, "y2": 8.0, "score": 0.5, "class_id": 0},
            {"x1": 16.0, "y1": 18.0, "x2": 80.0, "y2": 82.0, "score": 0.75, "class_id": 5}]
    if not omit_raw:
        manifest["raw_tensors"] = raw_entries
    if omit_hashes:
        manifest.pop("model_sha256")
        manifest.pop("image_sha256")
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return directory


def make_unified_run_record(capture_dir: Path, *, model: bytes, image: bytes,
                             unified_binary: bytes, soc: str = "S100",
                             rc: int = 0, role: str = "unified") -> None:
    (capture_dir / "stdout.txt").write_text("unified ran\n")
    (capture_dir / "stderr.txt").write_text("")
    record = {
        "schema": "rdk-model-zoo/yolov5-cpp-run-record/v1", "role": role,
        "utc_start": "2026-09-24T00:00:00Z", "utc_finish": "2026-09-24T00:00:04Z",
        "binary_path": "/tmp/yolov5_cpp",
        "binary_sha256_before": sha(unified_binary),
        "binary_sha256_after": sha(unified_binary),
        "model_path": "/tmp/m.hbm", "model_sha256_before": sha(model),
        "model_sha256_after": sha(model),
        "image_path": "/tmp/i.jpg", "image_sha256_before": sha(image),
        "image_sha256_after": sha(image),
        "cwd": "/tmp", "soc_name": soc, "board_soc": soc,
        "argv": ["/tmp/yolov5_cpp", "--dump-dir", "/tmp/dump"],
        "return_code": rc,
    }
    (capture_dir / "run-record.json").write_text(json.dumps(record), encoding="utf-8")


def run_compare(tmp: Path, source: Path, unified: Path, output: Path,
                target: str = "s100") -> subprocess.CompletedProcess:
    binary = tmp / "binaries"
    binary.mkdir(exist_ok=True)
    (binary / "source").write_bytes(b"source-binary")
    (binary / "unified").write_bytes(b"unified-binary")
    unified_record_dir = tmp / "unified-record"
    unified_record_dir.mkdir(exist_ok=True)
    make_unified_run_record(unified_record_dir, model=b"model-bytes",
                            image=b"image-bytes", unified_binary=b"unified-binary",
                            soc=target.upper())
    return subprocess.run(
        [PY, str(NATIVE / "compare_native.py"), "--target", target,
         "--repo-root", str(ROOT), "--source-capture", str(source),
         "--unified-dump", str(unified), "--source-binary", str(binary / "source"),
         "--unified-binary", str(binary / "unified"),
         "--unified-run-record", str(unified_record_dir / "run-record.json"),
         "--output", str(output)],
        capture_output=True, text=True)


class InstrumentTests(unittest.TestCase):
    def test_generates_pinned_sources_with_unique_anchors(self):
        for target, expected_sources, expected_anchors in (("x5", 1, 10), ("s100", 2, 9)):
            with tempfile.TemporaryDirectory() as tmp:
                work = Path(tmp) / f"work-{target}"
                result = subprocess.run(
                    [PY, str(NATIVE / "instrument.py"), "--target", target,
                     "--repo-root", str(ROOT), "--work-dir", str(work),
                     "--model-path", "/tmp/m.bin", "--image-path", "/tmp/bus.jpg"],
                    capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                audit = json.loads((work / "instrumentation-audit.json").read_text())
                self.assertEqual(len(audit["sources"]), expected_sources)
                for source in audit["sources"]:
                    self.assertEqual(source["pinned_sha256"], source["blob_sha256"])
                    for anchor in source["anchors"]:
                        self.assertEqual(anchor["matched"], 1, anchor)
                self.assertEqual(sum(len(s["anchors"]) for s in audit["sources"]),
                                 expected_anchors)
                for path in work.rglob("*.cc"):
                    if path.name == "main.cc":
                        self.assertIn("ycap::", path.read_text())
                for path in (work / "src").glob("*.cpp"):
                    self.assertIn("ycap::", path.read_text())
                if target == "x5":
                    main = (work / "main.cc").read_text()
                    self.assertIn('MODEL_PATH "/tmp/m.bin"', main)
                    rewrites = {item["macro"] for item in audit["path_rebindings"]}
                    self.assertEqual(rewrites, {"MODEL_PATH", "TESR_IMG_PATH"})
                else:
                    closure = {item["source_path"] for item in audit["closure"]}
                    self.assertIn("utils/c_utils/src/postprocess.cpp", closure)
                    self.assertIn(
                        "samples/vision/yolov5/runtime/cpp/inc/yolov5.hpp", closure)
                    for item in audit["closure"]:
                        copy = work / item["work_copy"]
                        self.assertTrue(copy.is_file(), item["work_copy"])
                        self.assertEqual(sha(copy.read_bytes()), item["blob_sha256"])
                again = subprocess.run(
                    [PY, str(NATIVE / "instrument.py"), "--target", target,
                     "--repo-root", str(ROOT), "--work-dir", str(work)],
                    capture_output=True, text=True)
                self.assertEqual(again.returncode, 2, again.stderr)

    def test_shallow_clone_missing_commit_gives_preparation_hint(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            subprocess.run(["git", "init", "-q", str(repo)], check=True)
            (repo / "README").write_text("x\n")
            subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
            subprocess.run(["git", "-C", str(repo), "-c", "user.email=t@t",
                            "-c", "user.name=t", "commit", "-qm", "x"], check=True)
            result = subprocess.run(
                [PY, str(NATIVE / "instrument.py"), "--target", "x5",
                 "--repo-root", str(repo), "--work-dir", str(Path(tmp) / "w")],
                capture_output=True, text=True)
            self.assertEqual(result.returncode, 3)
            self.assertIn("fetch --depth=1", result.stderr)
            self.assertIn("working-tree files are never accepted", result.stderr)

    def test_anchor_drift_fails_closed(self):
        instrument = load_module("b7_instrument", NATIVE / "instrument.py")
        text = "alpha\nbeta\ngamma\n"
        out, applied = instrument.apply_anchors(text, [("a1", "beta", "INSERTED\n")])
        self.assertIn("INSERTED", out)
        with self.assertRaises(instrument.InstrumentationError):
            instrument.apply_anchors(text, [("a1", "missing", "x\n")])
        with self.assertRaises(instrument.InstrumentationError):
            instrument.apply_anchors("beta\nbeta\n", [("a1", "beta", "x\n")])

    def test_wrong_pin_fails_closed(self):
        instrument = load_module("b7_instrument", NATIVE / "instrument.py")
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            subprocess.run(["git", "init", "-q", str(repo)], check=True)
            (repo / "file.cc").write_text("int main(){}\n")
            subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
            subprocess.run(["git", "-C", str(repo), "-c", "user.email=t@t",
                            "-c", "user.name=t", "commit", "-qm", "x"], check=True)
            with self.assertRaises(instrument.InstrumentationError):
                instrument.pinned_blob(repo, "HEAD", "file.cc", "0" * 64)


class ObserverTests(unittest.TestCase):
    def _compile(self, tmp: Path, body: str) -> Path:
        probe = tmp / "probe.cc"
        probe.write_text('#include "ycap_observer.hpp"\nint main(int argc, char** argv) {\n'
                         '    (void)argc; (void)argv;\n' + body +
                         "    return 0;\n}\n")
        binary = tmp / "probe"
        build = subprocess.run(
            ["c++", "-std=c++17", "-Wall", "-Wextra", "-I", str(NATIVE),
             str(probe), "-o", str(binary)], capture_output=True, text=True)
        self.assertEqual(build.returncode, 0, build.stderr)
        return binary

    def test_generated_x5_hook_order_produces_a_valid_capture(self):
        """Runs the ACTUAL hook sequence of the generated X5 main.cc (review B).

        instrument.py generates the file; every ycap call is extracted in
        textual order with its tensor argument, replayed against the real
        observer in that order, and the capture must succeed — in particular
        tensor_meta("input0") must precede payload("input0")."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            work = root / "work"
            generated = subprocess.run(
                [PY, str(NATIVE / "instrument.py"), "--target", "x5",
                 "--repo-root", str(ROOT), "--work-dir", str(work),
                 "--model-path", "/tmp/m.bin", "--image-path", "/tmp/bus.jpg"],
                capture_output=True, text=True)
            self.assertEqual(generated.returncode, 0, generated.stderr)
            text = (work / "main.cc").read_text()
            calls = re.findall(
                r'ycap::(\w+)\(\s*(?:\(\s*std::string\()?"([^"]*)"', text)
            sequence = [(function, argument) for function, argument in calls]
            self.assertGreaterEqual(len(sequence), 9, sequence)
            # The order constraint from the independent review: the input
            # metadata precedes the input payload, and all output metadata
            # precedes all output payloads.
            order = [f"{f}:{a}" for f, a in sequence]
            self.assertLess(order.index("tensor_meta:input0"),
                            order.index("payload:input0"))
            self.assertLess(max(i for i, entry in enumerate(order)
                                if entry == "tensor_meta:output"),
                            min(i for i, entry in enumerate(order)
                                if entry == "payload:output"))
            # Replay the extracted sequence verbatim against the observer.
            model_file = root / "m.bin"
            model_file.write_bytes(b"model-bytes")
            image_file = root / "bus.jpg"
            image_file.write_bytes(b"image-bytes")
            lines = ['    ycap::begin("driver");',
                     f'    ycap::note_path("model", "{model_file}");',
                     f'    ycap::note_path("image", "{image_file}");',
                     '    ycap::note_thresholds(0.25, 0.45);']
            # The generated names for the three heads are all built from
            # std::string("output") + index; disambiguate by occurrence.
            # The output hooks live inside source loops, so one textual call
            # stands for three runtime invocations: metas and payloads index
            # the same counter space, and the meta loop precedes the payload
            # loop in the generated program order.
            meta_counts: dict[str, int] = {}
            payload_counts: dict[str, int] = {}
            payload_total = {}
            for function, argument in sequence:
                if function == "payload" and argument in ("input", "output"):
                    payload_total[argument] = payload_total.get(argument, 0) + 1
            emit = []
            seen_meta = set()
            for function, argument in sequence:
                if function in ("begin", "note_path", "note_thresholds"):
                    continue
                name = argument
                if function == "tensor_meta" and argument in ("input", "output"):
                    # One textual meta inside a loop covers every payload of
                    # that family, so emit as many metas as payloads follow.
                    index = meta_counts.get(argument, 0)
                    total = payload_total.get(argument, 1)
                    for offset in range(max(1, total)):
                        emit.append((f"{argument}{index + offset}", 1 if argument == "input" else 0))
                    meta_counts[argument] = index + max(1, total)
                    for emit_name, emit_inputs in emit:
                        lines.append(
                            f'    ycap::tensor_meta("{emit_name}", "uint8", 1, 3, 2, 2, '
                            f'12, 6, 2, 1, 12, "none", 0, 0, -1, {emit_inputs});')
                        seen_meta.add(emit_name)
                    emit.clear()
                    continue
                elif function == "payload" and argument in ("input", "output"):
                    index = payload_counts.get(argument, 0)
                    payload_counts[argument] = index + 1
                    name = f"{argument}{index}"
                if function == "tensor_meta":
                    inputs = 1 if name.startswith("input") else 0
                    lines.append(
                        f'    ycap::tensor_meta("{name}", "uint8", 1, 3, 2, 2, '
                        f'12, 6, 2, 1, 12, "none", 0, 0, -1, {inputs});')
                    seen_meta.add(name)
                    continue
                elif function == "payload":
                    self.assertIn(name, seen_meta,
                                  f"payload({name}) fired before its metadata")
                    lines.append(f'    ycap::payload("{name}", buffer, 12);')
                elif function == "tensor_scale":
                    lines.append(f'    ycap::tensor_scale("{name}", scales, 1);')
                elif function in ("detection", "detection_original"):
                    lines.append(f'    ycap::{function}(1, 2, 3, 4, 0.5, 7);')
                elif function.startswith("tensor_zero"):
                    lines.append(f'    ycap::{function}("{name}", zeros, 1);')
                elif function == "finish":
                    lines.append("    ycap::finish(0);")
            # detection calls carry no string argument, so the extractor
            # cannot see them; emit one of each (their order is not the
            # property under test here).
            lines.append("    ycap::detection(1, 2, 3, 4, 0.5, 7);")
            lines.append("    ycap::detection_original(9, 8, 7, 6, 0.5, 7);")
            lines.append("    ycap::finish(0);")
            body = ("    static unsigned char buffer[12] = {0};\n"
                    "    static float scales[1] = {0.5f};\n"
                    "    static int zeros[1] = {0};\n" + "\n".join(lines) + "\n")
            binary = self._compile(root, body)
            capture_dir = root / "cap"
            import os
            run = subprocess.run([str(binary)], capture_output=True, text=True,
                                 env={**os.environ,
                                      "YOLOV5_CAPTURE_DIR": str(capture_dir)})
            self.assertEqual(run.returncode, 0, run.stderr)
            manifest = json.loads((capture_dir / "capture.json").read_text())
            self.assertFalse(manifest["failed"],
                             f"observer error: {manifest['error']!r}")
            self.assertTrue(manifest["inputs"], "input metadata must be recorded")
            self.assertEqual(len(manifest["outputs"]), 3)
            self.assertEqual(len(manifest["detections"]), 1)
            self.assertEqual(len(manifest["detections_original"]), 1)
            self.assertFalse((capture_dir / "capture-in-progress.json").exists())

    def test_observer_roundtrip_precision_markers_and_refusal(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            binary = self._compile(root, (
                '    ycap::begin("probe");\n'
                '    ycap::note_path("model", getenv("PROBE_MODEL"));\n'
                '    ycap::note_thresholds(0.25, F32NMS);\n'
                '    ycap::tensor_meta("output0", "int32", 1, 2, 2, 3,\n'
                '                      32, 32, 16, 4, 32, "scale", 3, 0, 3, 0);\n'
                '    float scale[3] = {F32VAL, 0.25f, 0.123456789f};\n'
                '    ycap::tensor_scale("output0", scale, 3);\n'
                '    ycap::payload("output0", scale, 12);\n'
                '    ycap::detection(1.2345678901234, 2, 3, 4, 0.5, 7);\n'
                '    ycap::detection_original(9.8765432109876, 2, 3, 4, 0.5, 7);\n'
                '    ycap::finish(0);\n').replace(
                    "F32NMS", repr(F32(0.45))).replace(
                    "F32VAL", repr(F32(0.00418911874294281))))
            model = root / "model.bin"
            model.write_bytes(b"model-bytes")
            capture_dir = root / "cap"
            import os
            env = {**os.environ, "YOLOV5_CAPTURE_DIR": str(capture_dir),
                   "PROBE_MODEL": str(model)}
            run = subprocess.run([str(binary)], capture_output=True, text=True, env=env)
            self.assertEqual(run.returncode, 0, run.stderr)
            manifest = json.loads((capture_dir / "capture.json").read_text())
            self.assertFalse(manifest["failed"])
            self.assertEqual(manifest["model_sha256"], sha(b"model-bytes"))
            self.assertTrue(manifest["utc_start"] and manifest["utc_finish"])
            self.assertEqual(manifest["detections"][0]["x1"], 1.2345678901234)
            self.assertEqual(manifest["detections_original"][0]["x1"], 9.8765432109876)
            import numpy as np
            # float32 values roundtrip exactly through the 17-digit spelling.
            self.assertEqual(np.float32(manifest["score_threshold"]), np.float32(0.25))
            self.assertEqual(np.float32(manifest["nms_threshold"]), np.float32(0.45))
            self.assertEqual(np.float32(manifest["outputs"][0]["scale_values"][0]),
                             np.float32(0.00418911874294281))
            payload = (capture_dir / "payload-output0.bin").read_bytes()
            record = manifest["outputs"][0]
            self.assertEqual(record["payload_bytes"], len(payload))
            self.assertEqual(record["payload_sha256"], sha(payload))
            self.assertFalse((capture_dir / "capture-in-progress.json").exists())
            again = subprocess.run([str(binary)], capture_output=True, text=True, env=env)
            self.assertEqual(again.returncode, 0)
            self.assertIn("refused", (capture_dir / "capture-error.txt").read_text())

    def test_observer_noop_without_env(self):
        with tempfile.TemporaryDirectory() as tmp:
            binary = self._compile(Path(tmp), '    ycap::begin("p"); ycap::finish(0);\n')
            result = subprocess.run([str(binary)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0)
            self.assertFalse((Path(tmp) / "cap").exists())


class RunCaptureTests(unittest.TestCase):
    def test_runner_records_process_evidence_and_verifies_audit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            work = root / "work"
            generated = subprocess.run(
                [PY, str(NATIVE / "instrument.py"), "--target", "s100",
                 "--repo-root", str(ROOT), "--work-dir", str(work)],
                capture_output=True, text=True)
            self.assertEqual(generated.returncode, 0, generated.stderr)
            binary = root / "fake"
            binary.write_bytes(
                "#!/bin/sh\necho out-loud\necho err-loud >&2\n"
                'mkdir -p "$YOLOV5_CAPTURE_DIR"\nexit 0\n'.encode())
            binary.chmod(0o755)
            model, image = root / "m.hbm", root / "i.jpg"
            model.write_bytes(b"model-bytes")
            image.write_bytes(b"image-bytes")
            capture_dir = root / "cap"
            result = subprocess.run(
                [PY, str(NATIVE / "run_capture.py"), "--binary", str(binary),
                 "--capture-dir", str(capture_dir), "--model", str(model),
                 "--image", str(image), "--audit", str(work / "instrumentation-audit.json"),
                 "--", "--user-flag", "value"],
                capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            record = json.loads((capture_dir / "run-record.json").read_text())
            self.assertEqual(record["return_code"], 0)
            self.assertEqual(record["argv"],
                             [str(binary.resolve()), "--user-flag", "value"])
            self.assertTrue(record["audit_verification"]["passed"])
            self.assertEqual(record["binary_sha256_before"],
                             record["binary_sha256_after"])
            self.assertEqual((capture_dir / "stdout.txt").read_text(), "out-loud\n")
            self.assertEqual((capture_dir / "stderr.txt").read_text(), "err-loud\n")
            # A nonzero process rc propagates and is recorded.
            binary.write_bytes(b"#!/bin/sh\nexit 3\n")
            binary.chmod(0o755)
            capture2 = root / "cap2"
            result2 = subprocess.run(
                [PY, str(NATIVE / "run_capture.py"), "--binary", str(binary),
                 "--capture-dir", str(capture2), "--model", str(model),
                 "--image", str(image), "--audit", str(work / "instrumentation-audit.json")],
                capture_output=True, text=True)
            self.assertEqual(result2.returncode, 2)
            record2 = json.loads((capture2 / "run-record.json").read_text())
            self.assertEqual(record2["return_code"], 3)
            # A tampered instrumented file fails the audit verification.
            (work / "src" / "yolov5.cpp").write_text("// tampered\n")
            capture3 = root / "cap3"
            result3 = subprocess.run(
                [PY, str(NATIVE / "run_capture.py"), "--binary", str(binary),
                 "--capture-dir", str(capture3), "--model", str(model),
                 "--image", str(image), "--audit", str(work / "instrumentation-audit.json")],
                capture_output=True, text=True)
            self.assertEqual(result3.returncode, 2)
            record3 = json.loads((capture3 / "run-record.json").read_text())
            self.assertFalse(record3["audit_verification"]["passed"])
            self.assertIn("mismatch", record3["audit_verification"]["error"])
            # Audit failure must NOT execute the binary.
            self.assertIsNone(record3["return_code"])
            self.assertEqual((capture3 / "stdout.txt").read_text(), "")


class RunCaptureRound3Tests(unittest.TestCase):
    def test_empty_or_truncated_audit_rejected_without_running(self):
        for content in ("{}", "null", '{"schema": "wrong"}'):
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                work = root / "work"
                subprocess.run(
                    [PY, str(NATIVE / "instrument.py"), "--target", "s100",
                     "--repo-root", str(ROOT), "--work-dir", str(work)],
                    capture_output=True, check=True)
                (work / "instrumentation-audit.json").write_text(content)
                binary = root / "probe"
                binary.write_bytes("#!/bin/sh\necho SHOULD-NOT-RUN\nexit 0\n".encode())
                binary.chmod(0o755)
                model, image = root / "m.hbm", root / "i.jpg"
                model.write_bytes(b"m"); image.write_bytes(b"i")
                capture = root / "cap"
                result = subprocess.run(
                    [PY, str(NATIVE / "run_capture.py"), "--binary", str(binary),
                     "--capture-dir", str(capture), "--model", str(model),
                     "--image", str(image),
                     "--audit", str(work / "instrumentation-audit.json")],
                    capture_output=True, text=True)
                self.assertEqual(result.returncode, 2, content)
                record = json.loads((capture / "run-record.json").read_text())
                self.assertFalse(record["audit_verification"]["passed"])
                self.assertIsNone(record["return_code"])
                self.assertNotIn("SHOULD-NOT-RUN", (capture / "stdout.txt").read_text())

    def test_missing_closure_or_tampered_cmake_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            work = root / "work"
            subprocess.run(
                [PY, str(NATIVE / "instrument.py"), "--target", "s100",
                 "--repo-root", str(ROOT), "--work-dir", str(work)],
                capture_output=True, check=True)
            audit = json.loads((work / "instrumentation-audit.json").read_text())
            # Drop one closure entry from the audit: incomplete closure.
            audit["closure"] = audit["closure"][:-1]
            (work / "instrumentation-audit.json").write_text(json.dumps(audit))
            binary = root / "probe"
            binary.write_bytes(b"#!/bin/sh\nexit 0\n")
            binary.chmod(0o755)
            model, image = root / "m.hbm", root / "i.jpg"
            model.write_bytes(b"m"); image.write_bytes(b"i")
            result = subprocess.run(
                [PY, str(NATIVE / "run_capture.py"), "--binary", str(binary),
                 "--capture-dir", str(root / "cap1"), "--model", str(model),
                 "--image", str(image),
                 "--audit", str(work / "instrumentation-audit.json")],
                capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            record = json.loads((root / "cap1" / "run-record.json").read_text())
            self.assertIn("expected", record["audit_verification"]["error"])
            # Tamper the generated CMake after generation.
            subprocess.run(
                [PY, str(NATIVE / "instrument.py"), "--target", "s100",
                 "--repo-root", str(ROOT), "--work-dir", str(root / "work2")],
                capture_output=True, check=True)
            (root / "work2" / "CMakeLists.txt").write_text("# tampered\n")
            result2 = subprocess.run(
                [PY, str(NATIVE / "run_capture.py"), "--binary", str(binary),
                 "--capture-dir", str(root / "cap2"), "--model", str(model),
                 "--image", str(image),
                 "--audit", str(root / "work2" / "instrumentation-audit.json")],
                capture_output=True, text=True)
            self.assertEqual(result2.returncode, 2)
            self.assertIn("CMake", result2.stderr)

    def test_unified_role_records_process_evidence_without_audit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            binary = root / "unified"
            binary.write_bytes("#!/bin/sh\necho unified-out\nexit 0\n".encode())
            binary.chmod(0o755)
            model, image = root / "m.hbm", root / "i.jpg"
            model.write_bytes(b"m"); image.write_bytes(b"i")
            capture = root / "cap"
            result = subprocess.run(
                [PY, str(NATIVE / "run_capture.py"), "--role", "unified",
                 "--binary", str(binary), "--capture-dir", str(capture),
                 "--model", str(model), "--image", str(image),
                 "--", "--dump-dir", str(root / "dump")],
                capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            record = json.loads((capture / "run-record.json").read_text())
            self.assertEqual(record["role"], "unified")
            self.assertEqual(record["return_code"], 0)
            self.assertEqual(record["argv"],
                             [str(binary.resolve()), "--dump-dir", str(root / "dump")])
            self.assertEqual((capture / "stdout.txt").read_text(), "unified-out\n")


class CompareNativeTests(unittest.TestCase):
    def test_padded_layout_passes_and_writes_arrays(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            source = build_s100_capture(base / "source")
            unified = build_unified_dump(base / "unified")
            output = base / "out"
            result = run_compare(base, source, unified, output)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            report = json.loads((output / "comparison.json").read_text())
            self.assertTrue(report["passed"], report["failures"])
            stages = {entry["stage"]: entry["passed"] for entry in report["stages"]}
            self.assertTrue(stages["input:input0"])
            self.assertTrue(stages["raw:output0->output0"])
            self.assertTrue(stages["scale:output0"])
            self.assertTrue(stages["model-space:boxes"])
            self.assertTrue(stages["final-coordinates:boxes"])
            for name in ("input-input0-source.npy", "raw-output0-source.npy",
                         "detections-final-coordinates-unified.npy",
                         "originals/source-capture.json", "originals/unified-manifest.json",
                         "originals/source-run-record.json", "originals/source-stdout.txt",
                         "originals/raw-output0-source-physical.bin", "digests.json"):
                self.assertTrue((output / name).is_file(), name)

    def test_real_v2_manifest_parameters_schema_is_accepted(self):
        """Regression against the REAL board manifest (checked-in copy)."""
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            unified = base / "unified"
            unified.mkdir()
            shutil.copyfile(REAL_MANIFEST, unified / "manifest.json")
            source = build_s100_capture(base / "source")
            result = run_compare(base, source, unified, base / "out", target="x5")
            combined = result.stdout + result.stderr
            self.assertNotIn("parameters must be a JSON object", combined)
            self.assertEqual(result.returncode, 2)
            report = json.loads((base / "out" / "comparison.json").read_text())
            self.assertFalse(report["passed"])
            self.assertTrue(report["failures"])

    def test_board_identity_exact_alias_contract(self):
        compare = load_module("b7_compare", NATIVE / "compare_native.py")
        # Prefix matches are not identity.
        self.assertIn("conflicts", compare.board_conflict("s100", [("soc", "S100P")]))
        self.assertIn("not a known alias",
                      compare.board_conflict("s100", [("soc", "S100Whatever")]))
        self.assertIn("conflicts", compare.board_conflict("s100", [("soc", "S600")]))
        self.assertIsNone(compare.board_conflict("s100", [("soc", "S100")]))
        # socinfo aliases resolve (X5 boards report X5U, no boardinfo soc).
        self.assertIsNone(compare.board_conflict("x5", [("socinfo", "X5U")]))
        self.assertIsNone(compare.board_conflict("x5", [("socinfo", "X5M")]))
        self.assertIn("conflicts", compare.board_conflict("s100p", [("soc", "s100")]))
        # All readings empty is not identity either.
        self.assertIn("unrecorded", compare.board_conflict("s100", []))

    def test_unified_run_record_required_and_bound(self):
        # Missing unified record file.
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            source = build_s100_capture(base / "source")
            unified = build_unified_dump(base / "unified")
            binary = base / "b"
            binary.mkdir()
            (binary / "source").write_bytes(b"source-binary")
            (binary / "unified").write_bytes(b"unified-binary")
            result = subprocess.run(
                [PY, str(NATIVE / "compare_native.py"), "--target", "s100",
                 "--repo-root", str(ROOT), "--source-capture", str(source),
                 "--unified-dump", str(unified), "--source-binary",
                 str(binary / "source"), "--unified-binary", str(binary / "unified"),
                 "--unified-run-record", str(base / "missing.json"),
                 "--output", str(base / "out")], capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            self.assertIn("unified run record", result.stderr)
        # Wrong role / nonzero rc / board conflict on the unified side.
        for mutate, message in (
                (lambda r: r.update({"role": "source"}), "role is 'source'"),
                (lambda r: r.update({"return_code": 7}), "return_code=7"),
                (lambda r: r.update({"soc_name": "S600", "board_soc": "S600"}),
                 "unified side")):
            with tempfile.TemporaryDirectory() as tmp:
                base = Path(tmp)
                source = build_s100_capture(base / "source")
                unified = build_unified_dump(base / "unified")
                record_dir = base / "ur"
                record_dir.mkdir()
                make_unified_run_record(record_dir, model=b"model-bytes",
                                        image=b"image-bytes",
                                        unified_binary=b"unified-binary")
                record = json.loads((record_dir / "run-record.json").read_text())
                mutate(record)
                (record_dir / "run-record.json").write_text(json.dumps(record))
                binary = base / "b"
                binary.mkdir()
                (binary / "source").write_bytes(b"source-binary")
                (binary / "unified").write_bytes(b"unified-binary")
                result = subprocess.run(
                    [PY, str(NATIVE / "compare_native.py"), "--target", "s100",
                     "--repo-root", str(ROOT), "--source-capture", str(source),
                     "--unified-dump", str(unified), "--source-binary",
                     str(binary / "source"), "--unified-binary", str(binary / "unified"),
                     "--unified-run-record", str(record_dir / "run-record.json"),
                     "--output", str(base / "out")], capture_output=True, text=True)
                self.assertEqual(result.returncode, 2, message)
                self.assertIn(message, result.stderr)

    def test_restore_channel_stride_boundaries(self):
        compare = load_module("b7_compare", NATIVE / "compare_native.py")
        # s3 < itemsize overlaps channels.
        with self.assertRaises(compare.ComparisonError):
            compare.restore_logical(bytes(64), "int32", [1, 2, 2, 3], [64, 32, 16, 2])
        # s3 not a multiple of itemsize misaligns elements.
        with self.assertRaises(compare.ComparisonError):
            compare.restore_logical(bytes(64), "int32", [1, 2, 2, 3], [64, 32, 16, 6])
        # s2 < channels*s3 overlaps pixels.
        with self.assertRaises(compare.ComparisonError):
            compare.restore_logical(bytes(128), "int32", [1, 2, 2, 4], [128, 32, 12, 4])
        # Real supported layouts still decode: the published S layout family.
        data = bytes(7225344)
        restored = compare.restore_logical(data, "int32", [1, 84, 84, 255],
                                           [7225344, 86016, 1024, 4])
        self.assertEqual(restored.shape, (84 * 84 * 255,))
        # Channel padding (s3=8, aligned, >= itemsize) with a covering buffer.
        padded = compare.restore_logical(bytes(160), "int32", [1, 2, 2, 3],
                                         [160, 80, 40, 8])
        self.assertEqual(padded.shape, (12,))

    def test_board_identity_conflict_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            source = build_s100_capture(base / "source", soc="S600")
            unified = build_unified_dump(base / "unified")
            result = run_compare(base, source, unified, base / "out")
            self.assertEqual(result.returncode, 2)
            self.assertIn("conflicts with comparison target", result.stderr)

    def test_unified_payload_digests_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            source = build_s100_capture(base / "source")
            unified = build_unified_dump(base / "unified", drop_payload_hashes=True)
            result = run_compare(base, source, unified, base / "out")
            self.assertEqual(result.returncode, 2)
            self.assertTrue("lacks" in result.stderr, result.stderr)

    def test_unified_quantization_kind_must_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            source = build_s100_capture(base / "source")
            unified = build_unified_dump(base / "unified", flip_quanti=True)
            result = run_compare(base, source, unified, base / "out")
            self.assertEqual(result.returncode, 2)
            self.assertIn("quantization kind", result.stderr)

    def test_float32_threshold_and_scale_semantics(self):
        """0.45f vs "0.450000", 17- vs 9-digit scale spellings: same bits pass
        (covered by the standard pass test); different bits fail."""
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            source = build_s100_capture(base / "source")
            unified = build_unified_dump(base / "unified")
            manifest = json.loads((unified / "manifest.json").read_text())
            manifest["parameters"]["nms_thres"] = "0.449999"  # different float32
            (unified / "manifest.json").write_text(json.dumps(manifest))
            result = run_compare(base, source, unified, base / "out")
            self.assertEqual(result.returncode, 2)
            self.assertIn("float32-bit mismatch", result.stderr)
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            source = build_s100_capture(base / "source")
            unified = build_unified_dump(base / "unified", nine_digit_scales=False)
            manifest = json.loads((unified / "manifest.json").read_text())
            manifest["outputs"][0]["scale_values"][0] = 0.004189119  # different bits
            (unified / "manifest.json").write_text(json.dumps(manifest))
            result = run_compare(base, source, unified, base / "out")
            self.assertEqual(result.returncode, 2)
            self.assertIn("scale descriptor of output0 differs",
                          result.stdout + result.stderr)

    def test_every_failure_keeps_a_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            source = base / "source"
            source.mkdir()  # no capture.json at all
            unified = build_unified_dump(base / "unified")
            output = base / "out"
            result = run_compare(base, source, unified, output)
            self.assertEqual(result.returncode, 2)
            report = output / "comparison.json"
            self.assertTrue(report.is_file(), "failure must still write a report")
            self.assertFalse(json.loads(report.read_text())["passed"])
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            source = build_s100_capture(base / "source")
            (source / "capture.json").write_text("{not json")
            unified = build_unified_dump(base / "unified")
            output = base / "out"
            result = run_compare(base, source, unified, output)
            self.assertEqual(result.returncode, 2)
            self.assertTrue((output / "comparison.json").is_file())

    def test_run_record_required_and_bound(self):
        # Missing run record.
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            source = build_s100_capture(base / "source", omit_run_record=True)
            unified = build_unified_dump(base / "unified")
            result = run_compare(base, source, unified, base / "out")
            self.assertEqual(result.returncode, 2)
            self.assertIn("source run record", result.stderr)
        # Nonzero recorded rc.
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            source = build_s100_capture(base / "source")
            record = json.loads((source / "run-record.json").read_text())
            record["return_code"] = 3
            (source / "run-record.json").write_text(json.dumps(record))
            unified = build_unified_dump(base / "unified")
            result = run_compare(base, source, unified, base / "out")
            self.assertEqual(result.returncode, 2)
            self.assertIn("return_code=3", result.stderr)
        # Binary changed during the run.
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            source = build_s100_capture(base / "source")
            record = json.loads((source / "run-record.json").read_text())
            record["binary_sha256_after"] = "0" * 64
            (source / "run-record.json").write_text(json.dumps(record))
            unified = build_unified_dump(base / "unified")
            result = run_compare(base, source, unified, base / "out")
            self.assertEqual(result.returncode, 2)
            self.assertIn("changed during the run", result.stderr)
        # Failed audit verification.
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            source = build_s100_capture(base / "source")
            record = json.loads((source / "run-record.json").read_text())
            record["audit_verification"] = {"passed": False, "error": "mismatch"}
            (source / "run-record.json").write_text(json.dumps(record))
            unified = build_unified_dump(base / "unified")
            result = run_compare(base, source, unified, base / "out")
            self.assertEqual(result.returncode, 2)
            self.assertIn("audit verification failed", result.stderr)
        # Unified manifest binary hash must match the attested binary.
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            source = build_s100_capture(base / "source")
            unified = build_unified_dump(base / "unified")
            manifest = json.loads((unified / "manifest.json").read_text())
            manifest["binary_sha256"] = "0" * 64
            (unified / "manifest.json").write_text(json.dumps(manifest))
            result = run_compare(base, source, unified, base / "out")
            self.assertEqual(result.returncode, 2)
            self.assertIn("unified binary hash mismatch", result.stderr)

    def test_restore_strictness(self):
        compare = load_module("b7_compare", NATIVE / "compare_native.py")
        # pitch smaller than a row.
        with self.assertRaises(compare.ComparisonError):
            compare.restore_rows(b"0123456789", [1, 2, 4, 1], [0, 3, 1, 1])
        # buffer shorter than the plane.
        with self.assertRaises(compare.ComparisonError):
            compare.restore_rows(b"0123", [1, 2, 4, 1], [0, 8, 1, 1])
        # stride[1] inconsistent with the uniform row pitch.
        with self.assertRaises(compare.ComparisonError):
            compare.restore_logical(bytes(64), "int32", [1, 2, 2, 3], [64, 40, 16, 4])
        # mixed strides are not a proven layout.
        with self.assertRaises(compare.ComparisonError):
            compare.restore_logical(bytes(64), "int32", [1, 2, 2, 3], [64, 32, 16, 0])
        # compact requires an exact-size payload.
        with self.assertRaises(compare.ComparisonError):
            compare.restore_logical(bytes(64), "int32", [1, 2, 2, 3], [0, 0, 0, 0])
        # strided restore still works for the published S layout family.
        data = bytes(7225344)
        restored = compare.restore_logical(data, "int32", [1, 84, 84, 255],
                                           [7225344, 86016, 1024, 4])
        self.assertEqual(restored.shape, (84 * 84 * 255,))

    def test_missing_detection_keys_rejected(self):
        for kwargs, message in (({"drop_detections_key": True}, "detections"),
                                ({"drop_original_key": True}, "detections_original")):
            with tempfile.TemporaryDirectory() as tmp:
                base = Path(tmp)
                source = build_s100_capture(base / "source", **kwargs)
                unified = build_unified_dump(base / "unified")
                result = run_compare(base, source, unified, base / "out")
                self.assertEqual(result.returncode, 2, message)
                self.assertIn(message, result.stderr)
        # One-sided detections_original on the unified side also blocks.
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            source = build_s100_capture(base / "source")
            unified = build_unified_dump(base / "unified", include_original_dets=False)
            result = run_compare(base, source, unified, base / "out")
            self.assertEqual(result.returncode, 2)
            self.assertIn("detections_original", result.stderr)

    def test_pairs_parameters_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            source = build_s100_capture(base / "source")
            unified = build_unified_dump(base / "unified", parameters_style="pairs")
            result = run_compare(base, source, unified, base / "out")
            self.assertEqual(result.returncode, 2)
            self.assertIn("parameters must be a JSON object", result.stderr)

    def test_empty_lists_cannot_bypass(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            source = build_s100_capture(base / "source")
            unified = build_unified_dump(base / "unified", empty_lists=True)
            result = run_compare(base, source, unified, base / "out")
            self.assertEqual(result.returncode, 2)
            self.assertIn("exactly 3 heads", result.stderr)

    def test_wrong_output_count_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            source = build_s100_capture(base / "source")
            unified = build_unified_dump(base / "unified", output_count=2)
            result = run_compare(base, source, unified, base / "out")
            self.assertEqual(result.returncode, 2)
            self.assertIn("exactly 3", result.stderr)

    def test_extra_output_shape_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            source = build_s100_capture(base / "source")
            unified = build_unified_dump(base / "unified", extra_output=True)
            result = run_compare(base, source, unified, base / "out")
            self.assertEqual(result.returncode, 2)
            self.assertIn("exactly 3", result.stderr)

    def test_dtype_mismatch_not_masked_by_float_cast(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            source = build_s100_capture(base / "source")
            unified = build_unified_dump(base / "unified", dtype_override="float32")
            result = run_compare(base, source, unified, base / "out")
            self.assertEqual(result.returncode, 2)
            self.assertIn("never masked by a float cast", result.stderr)

    def test_missing_unified_hashes_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            source = build_s100_capture(base / "source")
            unified = build_unified_dump(base / "unified", omit_hashes=True)
            result = run_compare(base, source, unified, base / "out")
            self.assertEqual(result.returncode, 2)
            self.assertIn("lacks an execution-time model/image hash", result.stderr)

    def test_model_hash_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            source = build_s100_capture(base / "source")
            unified = build_unified_dump(base / "unified", wrong_model=True)
            result = run_compare(base, source, unified, base / "out")
            self.assertEqual(result.returncode, 2)
            self.assertIn("model hash mismatch", result.stderr)

    def test_stale_capture_markers_rejected(self):
        for marker in ("capture-in-progress.json", "capture-error.txt"):
            with tempfile.TemporaryDirectory() as tmp:
                base = Path(tmp)
                source = build_s100_capture(base / "source", fail_marker=marker)
                unified = build_unified_dump(base / "unified")
                result = run_compare(base, source, unified, base / "out")
                self.assertEqual(result.returncode, 2, marker)
                self.assertIn(marker, result.stderr)

    def test_tampered_payload_sha_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            source = build_s100_capture(base / "source")
            payload = source / "payload-output0.bin"
            data = bytearray(payload.read_bytes())
            data[0] ^= 0xFF
            payload.write_bytes(bytes(data))
            unified = build_unified_dump(base / "unified")
            result = run_compare(base, source, unified, base / "out")
            self.assertEqual(result.returncode, 2)
            self.assertIn("SHA-256", result.stderr)

    def test_target_must_match_exactly(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            source = build_s100_capture(base / "source")
            unified = build_unified_dump(base / "unified", build_target="s600")
            result = run_compare(base, source, unified, base / "out")
            self.assertEqual(result.returncode, 2)
            self.assertIn("does not exactly match", result.stderr)
            unified_s600 = build_unified_dump(base / "unified-s600", build_target="s600")
            source_s600 = build_s100_capture(base / "source-s600", soc="S600")
            result = run_compare(base, source_s600, unified_s600,
                                 base / "out-s600", target="s600")
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_raw_mismatch_fails_with_full_arrays(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            source = build_s100_capture(base / "source", tamper_raw=True)
            unified = build_unified_dump(base / "unified")
            output = base / "out"
            result = run_compare(base, source, unified, output)
            self.assertEqual(result.returncode, 2, result.stdout)
            report = json.loads((output / "comparison.json").read_text())
            self.assertFalse(report["passed"])
            self.assertTrue(any("raw output" in failure for failure in report["failures"]))
            self.assertTrue((output / "raw-output0-source.npy").is_file())

    def test_failed_unified_run_never_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            source = build_s100_capture(base / "source")
            unified = build_unified_dump(base / "unified", rc=2)
            result = run_compare(base, source, unified, base / "out")
            self.assertEqual(result.returncode, 2)
            self.assertIn("unified run failed", result.stderr)

    def test_refuses_existing_output_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            source = build_s100_capture(base / "source")
            unified = build_unified_dump(base / "unified")
            output = base / "out"
            output.mkdir()
            (output / "sentinel").write_text("keep me")
            result = run_compare(base, source, unified, output)
            self.assertEqual(result.returncode, 2)
            self.assertIn("refusing to overwrite", result.stderr)
            self.assertEqual((output / "sentinel").read_text(), "keep me")


# Host stubs for compiling the INSTRUMENTED fixed sources and the SDK-free
# helper surface. They only make the injected glue compile on a host; they
# prove nothing about a real SDK or OpenCV build, which stays a board concern.
FIXED_SOURCE_STUBS = {
    "opencv2/core/types.hpp": "#pragma once\n#include \"mat.hpp\"\n",
    "opencv2/core.hpp": "#pragma once\n#include \"core/mat.hpp\"\n",
    "opencv2/core/mat.hpp": """\
#pragma once
#include <cstddef>
#include <string>
#include <vector>
namespace cv {
template <typename T> struct Rect_ { T x = 0, y = 0, width = 0, height = 0; Rect_() = default;
  Rect_(T x_, T y_, T w_, T h_) : x(x_), y(y_), width(w_), height(h_) {} };
typedef Rect_<int> Rect;
typedef Rect_<double> Rect2d;
struct Point { int x = 0, y = 0; Point() = default; Point(int x_, int y_) : x(x_), y(y_) {} };
struct Size { int width = 0, height = 0; Size() = default; Size(int w, int h) : width(w), height(h) {} };
struct Scalar { double v[4] = {0, 0, 0, 0}; Scalar() = default; Scalar(double s) { v[0] = v[1] = v[2] = v[3] = s; }
  Scalar(double s0, double s1, double s2) { v[0] = s0; v[1] = s1; v[2] = s2; v[3] = 0; }
  Scalar(double s0, double s1, double s2, double s3) { v[0] = s0; v[1] = s1; v[2] = s2; v[3] = s3; } };
class Mat {
 public:
  Mat() = default;
  Mat(int rows, int cols, int type) : rows(rows), cols(cols), type_(type) {}
  Mat(int rows, int cols, int type, const Scalar& s) : rows(rows), cols(cols), type_(type) { (void)s; }
  bool empty() const { return rows <= 0 || cols <= 0; }
  int channels() const { return 3; }
  int type() const { return type_; }
  template <typename T> T* ptr() { return reinterpret_cast<T*>(data); }
  template <typename T> const T* ptr() const { return reinterpret_cast<const T*>(data); }
  Mat operator()(const Rect& roi) const { (void)roi; return Mat(); }
  void copyTo(Mat& dst) const { dst = *this; }
  void create(int rows_, int cols_, int type__) { rows = rows_; cols = cols_; type_ = type__; }
  int rows = 0, cols = 0;
  unsigned char* data = nullptr;
 private:
  int type_ = 0;
};
}
#define CV_8UC1 0
#define CV_8UC3 16
#define CV_VERSION "stub"
""",
    "opencv2/imgcodecs.hpp": """\
#pragma once
#include "core/mat.hpp"
#include <string>
namespace cv { Mat imread(const std::string&, int flags = 1); bool imwrite(const std::string&, Mat&); }
""",
    "opencv2/imgproc.hpp": """\
#pragma once
#include "core/mat.hpp"
#include <string>
#include <vector>
namespace cv {
enum { BORDER_CONSTANT = 0, COLOR_BGR2YUV_I420 = 84, FONT_HERSHEY_SIMPLEX = 0, LINE_AA = 16 };
void resize(const Mat& src, Mat& dst, Size dsize, double fx = 0, double fy = 0, int interpolation = 1);
void cvtColor(const Mat& src, Mat& dst, int code);
void copyMakeBorder(const Mat& src, Mat& dst, int top, int bottom, int left, int right,
                    int borderType, const Scalar& value = Scalar());
void rectangle(Mat& img, Point p1, Point p2, const Scalar& color, int thickness = 1,
               int lineType = 0, int shift = 0);
void putText(Mat& img, const std::string& text, Point org, int fontFace, double fontScale,
             const Scalar& color, int thickness = 1, int lineType = 0, bool bottomLeftOrigin = false);
}
""",
    "opencv2/highgui.hpp": "#pragma once\n#include \"core/mat.hpp\"\n",
    "opencv2/opencv.hpp": """\
#pragma once
#include "core/mat.hpp"
#include "imgcodecs.hpp"
#include "imgproc.hpp"
#include "highgui.hpp"
""",
    "opencv2/dnn/dnn.hpp": """\
#pragma once
#include <vector>
namespace cv { namespace dnn {
template <typename T>
void NMSBoxes(const std::vector<T>& bboxes, const std::vector<float>& scores,
              float score_threshold, float nms_threshold, std::vector<int>& indices,
              float eta = 1.0F, int top_k = 0) { (void)bboxes; (void)scores;
  (void)score_threshold; (void)nms_threshold; (void)indices; (void)eta; (void)top_k; }
} }
""",
    "gflags/gflags.h": """\
#pragma once
#include <cstdint>
#include <string>
#define DEFINE_string(name, val, tip) inline std::string FLAGS_##name = val;
#define DEFINE_double(name, val, tip) inline double FLAGS_##name = val;
#define DEFINE_int32(name, val, tip) inline std::int32_t FLAGS_##name = val;
#define DEFINE_int64(name, val, tip) inline std::int64_t FLAGS_##name = val;
#define DEFINE_bool(name, val, tip) inline bool FLAGS_##name = val;
namespace gflags {
inline void ParseCommandLineFlags(int* argc, char*** argv, bool remove_flags) {
  (void)argc; (void)argv; (void)remove_flags; }
inline void SetUsageMessage(const std::string& usage) { (void)usage; }
inline std::string GetArgv() { return std::string(); }
}
""",
    "nlohmann/json.hpp": "#pragma once\nnamespace nlohmann { class json; }\n",
    "dnn/plugin/hb_dnn_layer.h": "#pragma once\n",
    "dnn/plugin/hb_dnn_plugin.h": "#pragma once\n",
    "dnn/hb_sys.h": "#pragma once\n#include \"dnn/hb_dnn.h\"\n",
}


class InstrumentedCompileTests(unittest.TestCase):
    """The instrumented fixed sources must compile with only the injected glue.

    Compiled against host stubs (SDK shapes from test_cpp_contract plus the
    opencv-lite/gflags stubs above) with -fsyntax-only. This proves the
    injected snippets reference real variables of the pinned sources; it is
    explicitly NOT evidence of a real SDK build."""

    @classmethod
    def setUpClass(cls):
        cls.cpp_contract = load_module("b7_cpp_contract", TEST_CPP)
        cls.compiler = shutil.which("c++")
        if cls.compiler is None:
            raise unittest.SkipTest("host C++ compiler unavailable")

    def _write_stubs(self, root: Path, stubs: dict) -> None:
        for relative, text in stubs.items():
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)

    def _syntax_check(self, source: Path, includes: list[Path]) -> None:
        command = [self.compiler, "-std=c++17", "-fsyntax-only"]
        for include in includes:
            command += ["-I", str(include)]
        command.append(str(source))
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0,
                         f"{source}:\n{result.stderr[:4000]}")

    def test_instrumented_x5_source_compiles(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            work = root / "work"
            generated = subprocess.run(
                [PY, str(NATIVE / "instrument.py"), "--target", "x5",
                 "--repo-root", str(ROOT), "--work-dir", str(work),
                 "--model-path", "/tmp/m.bin", "--image-path", "/tmp/bus.jpg"],
                capture_output=True, text=True)
            self.assertEqual(generated.returncode, 0, generated.stderr)
            stubs = root / "stubs"
            self._write_stubs(stubs, FIXED_SOURCE_STUBS)
            sdk = root / "sdk"
            self._write_stubs(sdk, self.cpp_contract.X5_STUBS)
            self._syntax_check(work / "main.cc", [stubs, sdk, work])

    def test_instrumented_s_sources_compile(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            work = root / "work"
            generated = subprocess.run(
                [PY, str(NATIVE / "instrument.py"), "--target", "s100",
                 "--repo-root", str(ROOT), "--work-dir", str(work)],
                capture_output=True, text=True)
            self.assertEqual(generated.returncode, 0, generated.stderr)
            stubs = root / "stubs"
            self._write_stubs(stubs, FIXED_SOURCE_STUBS)
            sdk = root / "sdk"
            self._write_stubs(sdk, self.cpp_contract.S100_STUBS)
            (sdk / "hobot" / "dnn" / "hb_dnn_ext.h").write_text("#pragma once\n")
            for status in ("hb_ucp_status.h", "hb_dnn_status.h"):
                (sdk / status).write_text("#pragma once\n")
                (sdk / "hobot" / status).write_text("#pragma once\n")
            self._syntax_check(work / "src" / "main.cpp",
                               [stubs, sdk, work, work / "utils" / "c_utils" / "inc",
                                work / "inc"])
            self._syntax_check(work / "src" / "yolov5.cpp",
                               [stubs, sdk, work, work / "utils" / "c_utils" / "inc",
                                work / "inc"])


if __name__ == "__main__":
    unittest.main()
