# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Build and run the C++ host math harness against the captured fixtures.

The exported inputs come from ``tests/fixtures/*.npz`` (real board-stage
captures for bus.jpg) plus deterministic synthetic cases; every expected
value is computed with the product Python runtime, which the unittest suite
itself pins against TorchVision references.  The C++ host build replays the
same CPU stages (anchor table, RPN selection, RoIAlign, PartB layouts,
decoding tail, letterbox geometry, NMS, sha256, descriptor table) and this
script compares the outputs — bit-exact where the arithmetic is
deterministic, and within 1-ulp-scale tolerances where ``exp()``/``softmax``
summation order can legitimately differ by one rounding.

Usage (repo root on sys.path, run from anywhere):

    python samples/vision/fasterrcnn/tests/cpp/run_host_checks.py

Exits 0 when every check passes; prints one line per check otherwise.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[5]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from samples.vision.fasterrcnn.runtime.python import detect as D  # noqa: E402

SAMPLE = ROOT / "samples" / "vision" / "fasterrcnn"
FIXTURES = SAMPLE / "tests" / "fixtures"
CPP = SAMPLE / "tests" / "cpp"
RUNTIME_INC = SAMPLE / "runtime" / "cpp" / "inc"
RUNTIME_SRC = SAMPLE / "runtime" / "cpp" / "src"
BYTETRACK = ROOT / "samples" / "vision" / "bytetrack" / "runtime" / "cpp"


def _write(path: Path, array: np.ndarray) -> None:
    path.write_bytes(np.ascontiguousarray(array).tobytes())


def _load(name: str) -> dict:
    return np.load(FIXTURES / name)


def export_inputs(in_dir: Path) -> dict:
    """Write the C++ harness inputs; return the expected outputs."""
    rpn = _load("rpn_stage.npz")
    roi = _load("roialign_stage.npz")
    tail = _load("tail_stage.npz")

    anchors = rpn["anchors"].astype(np.float32)
    _write(in_dir / "anchors.f32", anchors)          # compared via anchors_out

    rpn_obj = rpn["rpn_obj"].astype(np.float32).reshape(-1)
    rpn_deltas = rpn["rpn_deltas"].astype(np.float32)
    _write(in_dir / "rpn_obj.f32", rpn_obj)
    _write(in_dir / "rpn_deltas.f32", rpn_deltas)
    proposals_py, n_valid_py = D.rpn_select(rpn_obj, rpn_deltas, anchors)

    feat0 = roi["feat0"].astype(np.float32)
    rows = roi["rows"].astype(np.int64)
    roi_rois = roi["proposals"][rows].astype(np.float32)
    _write(in_dir / "feat0.f32", feat0)
    _write(in_dir / "roi_rois.f32", roi_rois)
    _write(in_dir / "roi_count.i64", np.array([len(rows)], dtype=np.int64))
    pooled_rows_py = D.roi_align_feat0(feat0, roi_rois)

    # Full padded static batch: fixture rows first, padding zero.
    pooled_full = np.zeros((D.NUM_PROPOSALS, 256, 7, 7), dtype=np.float32)
    pooled_full[:len(rows)] = pooled_rows_py
    _write(in_dir / "pooled_full.f32", pooled_full)
    bank_py = D.pack_bank(pooled_full)

    cls = tail["class_logits"].astype(np.float32)
    reg = tail["box_regression"].astype(np.float32)
    proposals = tail["proposals"].astype(np.float32)
    n_valid = int(tail["n_valid"])
    _write(in_dir / "cls.f32", cls)
    _write(in_dir / "reg.f32", reg)
    _write(in_dir / "proposals.f32", proposals)
    _write(in_dir / "tail_n_valid.i64", np.array([n_valid], dtype=np.int64))
    det_boxes_py, det_scores_py, det_labels_py = D.tail(cls, reg, proposals, n_valid)

    # Packed PartB outputs: the S layout transpose of cls/reg.
    cls_packed = cls.T.reshape(1, D.NUM_CLASSES, D.NUM_PROPOSALS, 1)
    reg_packed = reg.T.reshape(1, D.NUM_CLASSES * 4, D.NUM_PROPOSALS, 1)
    _write(in_dir / "cls_packed.f32", cls_packed)
    _write(in_dir / "reg_packed.f32", reg_packed)

    # Exact-tie tail: two candidates share one logit vector, so their person
    # softmax scores tie bit-identically in every runtime, but they decode to
    # boxes IoU ~0.92 apart; a third candidate sits at IoU 0.51 with one of
    # them and 0.48 with the other.  Stable-order NMS must keep the lower
    # flatten index in both runtimes, and the third candidate's survival
    # must follow from that winner (the cascade a one-ulp score change on a
    # near-tie flips).  Canvas boxes mirror a real 1280x720 clip event
    # (letterbox scale 0.25).
    tie_hi = np.full(D.NUM_CLASSES, -20.0, dtype=np.float32)
    tie_hi[0] = 0.0
    tie_hi[1] = 3.7
    tie_mid = tie_hi.copy()
    tie_mid[1] = 1.4
    tie_lo = np.full(D.NUM_CLASSES, -20.0, dtype=np.float32)
    tie_lo[0] = 5.0
    tie_lo[1] = -5.0
    cls_tie = np.tile(tie_lo, (D.NUM_PROPOSALS, 1))
    cls_tie[12] = tie_hi
    cls_tie[68] = tie_hi                 # identical vector -> exact score tie
    cls_tie[30] = tie_mid
    reg_tie = np.zeros((D.NUM_PROPOSALS, D.NUM_CLASSES * 4), dtype=np.float32)
    proposals_tie = np.tile(np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32),
                            (D.NUM_PROPOSALS, 1))
    proposals_tie[12] = np.array([237.385325, 52.724625, 291.397400, 179.796100],
                                 dtype=np.float32)
    proposals_tie[68] = np.array([237.643775, 49.751200, 293.853650, 178.802825],
                                 dtype=np.float32)
    proposals_tie[30] = np.array([257.788422, 54.557091, 300.160736, 167.394379],
                                 dtype=np.float32)
    _write(in_dir / "tie_cls.f32", cls_tie)
    _write(in_dir / "tie_reg.f32", reg_tie)
    _write(in_dir / "tie_proposals.f32", proposals_tie)
    _write(in_dir / "tie_n_valid.i64", np.array([D.NUM_PROPOSALS], dtype=np.int64))
    e_tie = np.exp(cls_tie - cls_tie.max(axis=1, keepdims=True))
    probs_tie = e_tie / e_tie.sum(axis=1, keepdims=True)
    if probs_tie[12, 1] != probs_tie[68, 1]:
        raise ValueError("tie fixture rows 12/68 do not score bit-identically")
    tie_boxes_py, tie_scores_py, tie_labels_py = D.tail(
        cls_tie, reg_tie, proposals_tie, D.NUM_PROPOSALS)
    decoded_tie = D.decode_deltas(proposals_tie, reg_tie, D.BOX_WEIGHTS).reshape(
        D.NUM_PROPOSALS, D.NUM_CLASSES, 4)
    if not (np.array_equal(tie_boxes_py[0], decoded_tie[12, 1])
            and np.array_equal(tie_boxes_py[1], decoded_tie[30, 1])):
        raise ValueError("tie fixture did not resolve to rows 12 and 30")

    # Letterbox: the bus geometry plus every detection mapped back.
    rows_img, cols_img = 1080, 810
    _write(in_dir / "letterbox_rows.i64", np.array([rows_img], dtype=np.int64))
    _write(in_dir / "letterbox_cols.i64", np.array([cols_img], dtype=np.int64))
    context_py = D.letterbox_geometry(rows_img, cols_img)
    _write(in_dir / "canvas_box_count.i64",
           np.array([len(det_boxes_py)], dtype=np.int64))
    _write(in_dir / "canvas_boxes.f32", det_boxes_py.astype(np.float32))
    original_py = context_py.to_original(det_boxes_py.astype(np.float32))

    # NMS: a deterministic case with ties and an overlap chain, plus the
    # float32 .7 boundary pair (double-arithmetic kernels wrongly drop box 1:
    # float32 IoU 14/20 rounds below the threshold, double IoU does not).
    nms_boxes = np.array([
        [10, 10, 50, 50], [11, 10, 51, 50], [12, 11, 52, 51],
        [100, 10, 140, 50], [10, 100, 50, 140],
    ], dtype=np.float32)
    nms_scores = np.array([0.9, 0.9, 0.8, 0.7, 0.6], dtype=np.float32)
    _write(in_dir / "nms_boxes.f32", nms_boxes)
    _write(in_dir / "nms_scores.f32", nms_scores)
    _write(in_dir / "nms_count.i64", np.array([len(nms_scores)], dtype=np.int64))
    _write(in_dir / "nms_thresh.f32", np.array([0.5], dtype=np.float32))
    keep_py = D.nms(nms_boxes, nms_scores, 0.5)

    boundary_boxes = np.array([[0, 0, 17, 1], [3, 0, 20, 1]], dtype=np.float32)
    boundary_scores = np.array([0.9, 0.8], dtype=np.float32)
    _write(in_dir / "boundary_boxes.f32", boundary_boxes)
    _write(in_dir / "boundary_scores.f32", boundary_scores)
    _write(in_dir / "boundary_count.i64", np.array([2], dtype=np.int64))
    _write(in_dir / "boundary_thresh.f32", np.array([0.7], dtype=np.float32))
    keep_boundary_py = D.nms(boundary_boxes, boundary_scores, 0.7)

    sha_bytes = bytes(range(256)) * 41 + b"fasterrcnn"
    (in_dir / "sha_input.bin").write_bytes(sha_bytes)
    sha_expected = hashlib.sha256(sha_bytes).hexdigest()

    descriptor = json.loads(
        (SAMPLE / "model" / "frcnn_mbv3_320.json").read_text())
    descriptor_lines = [
        "{target} {march} {pf} {psha} {bf} {bsha} {layout} {plat} {pblat}".format(
            target=target, march=row["march"],
            pf=row["parta"]["file"], psha=row["parta"]["sha256"],
            bf=row["partb"]["file"], bsha=row["partb"]["sha256"],
            layout=row["partb"]["layout"],
            plat=repr(float(row["compiler_latency_us"]["parta"])),
            pblat=repr(float(row["compiler_latency_us"]["partb"])))
        for target, row in descriptor["targets"].items()]

    return {
        "anchors": anchors.reshape(-1),
        "proposals": proposals_py.astype(np.float32),
        "n_valid": n_valid_py,
        "pooled_rows": pooled_rows_py.reshape(-1),
        "bank": bank_py.reshape(-1),
        "cls": cls.reshape(-1),
        "reg": reg.reshape(-1),
        "det_boxes": det_boxes_py,
        "det_scores": det_scores_py,
        "det_labels": det_labels_py,
        "tie_det_boxes": tie_boxes_py,
        "tie_det_scores": tie_scores_py,
        "tie_det_labels": tie_labels_py,
        "net": (context_py.net_size[0], context_py.net_size[1]),
        "scale": context_py.scale,
        "original": original_py,
        "keep": keep_py.astype(np.int32),
        "keep_boundary": keep_boundary_py.astype(np.int32),
        "sha256": sha_expected,
        "descriptor": descriptor_lines,
    }


def compile_harness(build_dir: Path) -> Path:
    """Compile host_checks + detect.cpp + the ByteTrack core (host build)."""
    compiler = shutil.which("c++") or shutil.which("clang++") or shutil.which("g++")
    if compiler is None:
        raise RuntimeError("no C++ compiler found for the host harness")
    binary = build_dir / "host_checks"
    sources = [
        str(CPP / "host_checks.cpp"),
        str(RUNTIME_SRC / "detect.cpp"),
        str(BYTETRACK / "src" / "tracking.cpp"),
        str(BYTETRACK / "src" / "lapjv.cpp"),
        str(BYTETRACK / "src" / "parallel.cpp"),
    ]
    cmd = [
        compiler, "-std=c++17", "-O2", "-ffp-contract=off", "-pthread",
        "-Wall", "-Wextra", "-Wno-unused-parameter",
        f"-I{RUNTIME_INC}", f"-I{BYTETRACK / 'inc'}",
        *sources, "-o", str(binary),
    ]
    subprocess.run(cmd, check=True,
                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return binary


RPN_PROPOSALS_ATOL = 1e-4  # px, absolute — same bound as the Python-side check


def proposals_match(actual: np.ndarray, expected: np.ndarray) -> bool:
    """Absolute 1e-4 px RPN-proposal tolerance (rtol 0, matching the
    Python fixture check): last-bit libm/numpy drift stays below 3.1e-5
    across implementations, while a wrong box, a reordering, or a
    non-finite coordinate exceeds 1e-4 outright (no relative term can
    widen the bound at large coordinates)."""
    return (actual.shape == expected.shape
            and bool(np.allclose(actual, expected, rtol=0.0,
                                 atol=RPN_PROPOSALS_ATOL)))


def compare(out_dir: Path, expected: dict) -> list[str]:
    """Compare the harness outputs with the Python-computed expectations."""
    failures: list[str] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        if ok:
            print(f"  PASS {name}")
        else:
            failures.append(f"{name}: {detail}")
            print(f"  FAIL {name}: {detail}")

    def load(name: str, dtype: np.dtype) -> np.ndarray:
        return np.fromfile(out_dir / name, dtype=dtype)

    # Deterministic double-math stages: bit-exact.
    check("anchor_table bit-exact",
          np.array_equal(load("anchors_out.f32", np.float32), expected["anchors"]),
          "differs from the fixture anchor table")
    # The RPN decode is exp() arithmetic like the tail below: identical
    # float32 math except that libm (C++ std::exp) and numpy's vectorized
    # exp round the last bits differently.  Measured on glibc: 76/600
    # coordinates differ, max abs 3.05e-05 (two float32 ULP at the 320-px
    # canvas scale), with cancellation at small coordinates keeping the
    # absolute drift below 1e-6.  Count and shape stay exact, and the
    # element-wise tolerance keeps the ordering checked — a swap moves
    # coordinates by whole pixels.
    proposals = load("proposals_out.f32", np.float32).reshape(-1, 4)
    check("rpn_select proposals match",
          proposals_match(proposals, expected["proposals"]),
          "proposals differ from the Python RPN selection")
    check("rpn_select n_valid",
          int(load("n_valid_out.i64", np.int64)[0]) == expected["n_valid"],
          f"n_valid differs from Python ({expected['n_valid']})")
    check("roi_align bit-exact",
          np.array_equal(load("pooled_out.f32", np.float32),
                         expected["pooled_rows"]),
          "pooled values differ from the Python RoIAlign")
    check("pack_bank bit-exact",
          np.array_equal(load("bank_out.f32", np.float32), expected["bank"]),
          "ROI bank layout differs from Python pack_bank")
    check("unpack_bank_outputs bit-exact",
          np.array_equal(load("cls_out.f32", np.float32), expected["cls"])
          and np.array_equal(load("reg_out.f32", np.float32), expected["reg"]),
          "unpacked PartB outputs differ from the source rows")

    # exp()-dependent stages: identical float32 arithmetic except that
    # libm/numpy may round the last bit differently.
    det_boxes = load("det_boxes_out.f32", np.float32).reshape(-1, 4)
    det_scores = load("det_scores_out.f32", np.float32)
    det_labels = load("det_labels_out.i64", np.int64)
    if det_labels.shape == expected["det_labels"].shape:
        boxes_ok = det_boxes.shape == expected["det_boxes"].shape and np.allclose(
            det_boxes, expected["det_boxes"], rtol=1e-5, atol=1e-4)
        scores_ok = det_scores.shape == expected["det_scores"].shape and np.allclose(
            det_scores, expected["det_scores"], rtol=1e-5, atol=1e-7)
        check("tail detections match", boxes_ok and scores_ok
              and np.array_equal(det_labels, expected["det_labels"]),
              f"{len(det_labels)} detections vs {len(expected['det_labels'])} expected")
    else:
        check("tail detections match", False,
              f"label count {len(det_labels)} != {len(expected['det_labels'])}")

    tie_boxes = load("tie_det_boxes_out.f32", np.float32).reshape(-1, 4)
    tie_scores = load("tie_det_scores_out.f32", np.float32)
    tie_labels = load("tie_det_labels_out.i64", np.int64)
    if tie_labels.shape == expected["tie_det_labels"].shape:
        tie_ok = (np.array_equal(tie_labels, expected["tie_det_labels"])
                  and tie_boxes.shape == expected["tie_det_boxes"].shape
                  and np.allclose(tie_boxes, expected["tie_det_boxes"],
                                  rtol=1e-5, atol=1e-4)
                  and np.allclose(tie_scores, expected["tie_det_scores"],
                                  rtol=1e-5, atol=1e-7))
        check("tail exact-tie NMS stability", tie_ok,
              "tied candidates must resolve to the lower flatten index and "
              "the overlapping third candidate must follow that winner")
    else:
        check("tail exact-tie NMS stability", False,
              f"label count {len(tie_labels)} != "
              f"{len(expected['tie_det_labels'])} expected")

    letterbox = (out_dir / "letterbox_out.txt").read_text().split()
    check("letterbox geometry",
          int(letterbox[0]) == expected["net"][0]
          and int(letterbox[1]) == expected["net"][1]
          and float(letterbox[2]) == expected["scale"],
          f"got {letterbox}, expected net={expected['net']} scale={expected['scale']!r}")
    check("to_original bit-exact",
          np.array_equal(load("original_out.f32", np.float32).reshape(-1, 4),
                         expected["original"]),
          "canvas->original mapping differs from Python")

    check("nms keep order",
          np.array_equal(load("keep_out.i32", np.int32), expected["keep"]),
          "kept indices differ from the Python NMS")
    check("nms .7 float32 boundary",
          np.array_equal(load("keep_boundary_out.i32", np.int32),
                         expected["keep_boundary"]),
          f"boundary pair kept {load('keep_boundary_out.i32', np.int32).tolist()}, "
          f"expected {expected['keep_boundary'].tolist()} (float32 IoU, strict >)")
    check("file_sha256",
          (out_dir / "sha256_out.txt").read_text().strip() == expected["sha256"],
          "streaming sha256 differs from hashlib")

    descriptor_out = [line.strip() for line in
                      (out_dir / "descriptor_out.txt").read_text().splitlines()
                      if line.strip()]
    # The latency doubles print with the stream's own precision; compare the
    # parsed values field by field instead of raw strings.
    check("descriptor table matches model/frcnn_mbv3_320.json",
          len(descriptor_out) == len(expected["descriptor"]) and all(
              _descriptor_line_equal(got, want)
              for got, want in zip(descriptor_out, expected["descriptor"])),
          f"got {descriptor_out!r}, expected {expected['descriptor']!r}")
    return failures


def _descriptor_line_equal(got: str, want: str) -> bool:
    g, w = got.split(), want.split()
    if len(g) != len(w) or len(g) != 9:
        return False
    for i in (0, 1, 2, 3, 4, 5, 6):    # target, march, files, hashes, layout
        if g[i] != w[i]:
            return False
    return all(float(a) == float(b) for a, b in zip(g[7:], w[7:]))


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="frcnn-host-checks-") as tmp:
        tmp_dir = Path(tmp)
        in_dir = tmp_dir / "in"
        out_dir = tmp_dir / "out"
        in_dir.mkdir()
        out_dir.mkdir()
        print("exporting fixtures and expectations ...")
        expected = export_inputs(in_dir)
        print("compiling the C++ host harness ...")
        binary = compile_harness(tmp_dir)
        print("running host_checks ...")
        proc = subprocess.run([str(binary), str(in_dir), str(out_dir)],
                              capture_output=True, text=True)
        if proc.returncode != 0:
            print(proc.stdout)
            print(proc.stderr, file=sys.stderr)
            return 1
        print(proc.stdout.strip())
        failures = compare(out_dir, expected)
        if failures:
            print(f"\n{len(failures)} check(s) failed:")
            for failure in failures:
                print(f"  - {failure}")
            return 1
        print("all host checks passed")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
