# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Host contract checks for the ByteTrack C++ runtime.

The tracker core is SDK-free (detections -> tracks), so unlike the YOLOv5
sample the whole production surface compiles and runs on a host: this test
builds the real sources (tracking.cpp, lapjv.cpp, parallel.cpp, cli.cpp,
main.cpp) with the system compiler in C++17 and exercises the behavioural
harness, the production replay CLI on a generated BTCPP1 file, and the
timing harness. The board video path (ByteTrackModel over the YOLOv5 API)
needs no SDK headers in its own translation units and is compile-checked
against the real yolov5 headers here; the BPU run itself stays a board
activity.
"""

from __future__ import annotations

import json
import struct
import subprocess
import tempfile
import unittest
from pathlib import Path

# Bounded subprocess budgets (seconds): a wedged compiler or a hung binary
# surfaces as a failed check instead of an unbounded test run. The behaviour
# harness carries its own 300 s watchdog on top.
COMPILE_TIMEOUT = 240
RUN_TIMEOUT = 120

ROOT = Path(__file__).resolve().parents[4]
CPP = ROOT / "samples" / "vision" / "bytetrack" / "runtime" / "cpp"
TESTS_CPP = ROOT / "samples" / "vision" / "bytetrack" / "tests" / "cpp"
YOLOV5_INC = ROOT / "samples" / "vision" / "yolov5" / "runtime" / "cpp" / "inc"
YOLOV5_DETECT = ROOT / "samples" / "vision" / "yolov5" / "runtime" / "cpp" / "src" / "detect.cpp"

REPLAY_MAGIC = b"BTCPP1\0\0"
RECORDS_HEADER = "frame\tlane\ttrack_id\tx1\ty1\tx2\ty2\tscore\n"


def build_production(tmp: Path, binary: str, sources: list[Path],
                     extra_flags: list[str] | None = None) -> tuple[Path, str]:
    """Compile the given production sources; return (binary path, stderr)."""
    target = tmp / binary
    result = subprocess.run(
        ["c++", "-std=c++17", "-Wall", "-Wextra", "-Wpedantic",
         "-I", str(CPP / "inc"), *(extra_flags or []), *map(str, sources), "-o", str(target)],
        capture_output=True, text=True, timeout=COMPILE_TIMEOUT)
    return target, result.stderr


def write_replay(path: Path, frames: list[list[tuple[float, float, float, float, float]]],
                 img_h: int = 1080, img_w: int = 1920) -> None:
    """Write the BTCPP1 little-endian float32 replay format."""
    with path.open("wb") as handle:
        handle.write(REPLAY_MAGIC)
        handle.write(struct.pack("<IIII", 1, len(frames), img_h, img_w))
        for dets in frames:
            handle.write(struct.pack("<I", len(dets)))
            for det in dets:
                handle.write(struct.pack("<5f", *det))


def person_stream(persons: int, frames: int) -> list[list[tuple[float, float, float, float, float]]]:
    """Integer-coordinate constant-velocity stream: 3 persons, some empty and
    low-score frames, all values exactly representable in float32."""
    out = []
    for f in range(frames):
        if f % 40 >= 38:  # two empty frames every 40
            out.append([])
            continue
        dets = []
        for p in range(persons):
            x = 100 + 160 * p + 2 * f
            y = 200 + 90 * p + f
            score = 0.2 if (f % 37 == 5) else 0.9  # periodic low-score dip
            dets.append((float(x), float(y), float(x + 60), float(y + 150), score))
        out.append(dets)
    return out


class ByteTrackCppContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.tmp = Path(cls._tmp.name)
        cls.checks, stderr = build_production(
            cls.tmp, "bytetrack_checks",
            [TESTS_CPP / "bytetrack_checks.cpp", CPP / "src" / "cli.cpp",
             CPP / "src" / "tracking.cpp", CPP / "src" / "lapjv.cpp",
             CPP / "src" / "parallel.cpp"])
        assert cls.checks.exists(), stderr
        cls.replay_cli, stderr = build_production(
            cls.tmp, "bytetrack_replay",
            [CPP / "src" / "main.cpp", CPP / "src" / "cli.cpp",
             CPP / "src" / "tracking.cpp", CPP / "src" / "lapjv.cpp",
             CPP / "src" / "parallel.cpp"])
        assert cls.replay_cli.exists(), stderr
        cls.bench, stderr = build_production(
            cls.tmp, "bytetrack_bench",
            [TESTS_CPP / "bytetrack_bench.cpp", CPP / "src" / "cli.cpp",
             CPP / "src" / "tracking.cpp", CPP / "src" / "lapjv.cpp",
             CPP / "src" / "parallel.cpp"])
        assert cls.bench.exists(), stderr
        # The component goldens include the production tracking.cpp directly,
        # so tracking.cpp itself must not be on the link line. The vendored
        # lap solver keeps its upstream unused-parameter warning suppressed,
        # exactly like the CMake build does for that one file.
        cls.components, stderr = build_production(
            cls.tmp, "bytetrack_components",
            [TESTS_CPP / "bytetrack_components.cpp", CPP / "src" / "lapjv.cpp",
             CPP / "src" / "parallel.cpp"],
            extra_flags=["-Wno-unused-parameter"])
        assert cls.components.exists(), stderr

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    # ------------------------------------------------------------- structure --

    def test_required_native_files_exist(self):
        for path in (
            CPP / "CMakeLists.txt",
            CPP / "run.sh",
            CPP / "inc" / "tracking.hpp",
            CPP / "inc" / "cli.hpp",
            CPP / "inc" / "parallel.hpp",
            CPP / "inc" / "lapjv.h",
            CPP / "src" / "tracking.cpp",
            CPP / "src" / "lapjv.cpp",
            CPP / "src" / "parallel.cpp",
            CPP / "src" / "cli.cpp",
            CPP / "src" / "main.cpp",
            CPP / "THIRD_PARTY_LICENSES.md",
            CPP / "licenses" / "lap-LICENSE",
            TESTS_CPP / "bytetrack_checks.cpp",
            TESTS_CPP / "bytetrack_bench.cpp",
        ):
            self.assertTrue(path.is_file(), path)

    def test_cmake_requires_an_explicit_target(self):
        cmake = (CPP / "CMakeLists.txt").read_text()
        # The target must be an explicit cache variable, never board probing.
        self.assertIn("BYTETRACK_TARGET", cmake)
        self.assertIn("FATAL_ERROR", cmake)
        self.assertNotIn("/sys/class", cmake)
        for token in ("host", "x5", "s100", "s100p", "s600"):
            self.assertIn(token, cmake)

    def test_production_headers_stay_cxx17(self):
        for header in ("tracking.hpp", "cli.hpp", "parallel.hpp", "lapjv.h"):
            text = (CPP / "inc" / header).read_text()
            self.assertNotIn("std::span", header + ": " + text)
            self.assertNotIn("#include <span>", header + ": " + text)

    def test_benchmark_flags_stay_out_of_the_production_cli(self):
        # Timing flags belong to the tests/ harness only; the delivered CLI
        # surface stays small.
        cli = (CPP / "src" / "cli.cpp").read_text()
        for flag in ("--warmup", "--repeats", "--csv", "--json"):
            self.assertNotIn(flag, cli)
        bench = (TESTS_CPP / "bytetrack_bench.cpp").read_text()
        for flag in ("--warmup", "--repeats", "--csv", "--json"):
            self.assertIn(flag, bench)

    def test_main_video_path_constructs_the_named_model_and_predicts(self):
        main_src = (CPP / "src" / "main.cpp").read_text()
        for token in (
            "bytetrack::ByteTrackModel::Options",
            "bytetrack::ByteTrackModel model(model_opt)",
            "model.predict(frame)",
        ):
            self.assertIn(token, main_src)
        # The model composes the YOLOv5 API inside tracking, not in main; the
        # entry carries no SDK/OpenCV includes (video IO lives in the CLI).
        self.assertNotIn("yolov5::Yolov5", main_src)
        self.assertNotIn("opencv2", main_src)
        tracking_src = (CPP / "src" / "tracking.cpp").read_text()
        self.assertIn("yolov5::Yolov5", tracking_src)

    def test_multi_lane_stress_stays_out_of_the_delivery_cli(self):
        # The delivery entry is one stream; multi-lane stress lives only in
        # the benchmark harness (tests/).
        cli = (CPP / "src" / "cli.cpp").read_text()
        main_src = (CPP / "src" / "main.cpp").read_text()
        self.assertNotIn("--lanes", cli)
        self.assertNotIn("lanes", main_src)
        bench = (TESTS_CPP / "bytetrack_bench.cpp").read_text()
        self.assertIn("--lanes", bench)

    def test_board_model_path_compiles_against_the_real_yolov5_api(self):
        # ByteTrackModel lives in tracking.cpp under the target macros; its
        # translation units need no SDK headers (the SDK types stay inside
        # yolov5's detect.cpp), so every board target syntax-checks on a host
        # against the shipped yolov5 headers.
        for macros in ("BYTETRACK_TARGET_X5=1 BYTETRACK_TARGET_NAME=\"x5\"",
                       "BYTETRACK_TARGET_S=1 BYTETRACK_TARGET_NAME=\"s100\"",
                       "BYTETRACK_TARGET_S=1 BYTETRACK_TARGET_NAME=\"s100p\"",
                       "BYTETRACK_TARGET_S=1 BYTETRACK_TARGET_NAME=\"s600\""):
            result = subprocess.run(
                ["c++", "-std=c++17", "-Wall", "-Wextra", "-Wpedantic", "-fsyntax-only",
                 *["-D" + d for d in macros.split()], "-I", str(CPP / "inc"),
                 "-I", str(YOLOV5_INC), str(CPP / "src" / "tracking.cpp")],
                capture_output=True, text=True, timeout=COMPILE_TIMEOUT)
            self.assertEqual(result.returncode, 0, macros + ": " + result.stderr)

    def test_cli_header_pimpl_classes_construct_where_impl_is_incomplete(self):
        # VideoSink/VideoReader keep OpenCV behind impl pointers. Any TU that
        # includes cli.hpp (no SDK/OpenCV headers) and default-constructs or
        # destroys the sinks must compile: an inline defaulted constructor or
        # destructor would instantiate ~unique_ptr<SinkImpl> while SinkImpl is
        # still incomplete and break every includer.
        source = self.tmp / "pimpl_include.cpp"
        source.write_text(
            '#include "cli.hpp"\n'
            "int main() {\n"
            "  bytetrack::VideoSink sink;\n"
            "  bytetrack::ByteTrackModel::Frame frame;\n"
            "  (void)frame;\n"
            "  return 0;\n"
            "}\n")
        for macros in ("BYTETRACK_TARGET_X5=1", "BYTETRACK_TARGET_S=1"):
            obj = self.tmp / f"pimpl_include-{macros.split('=')[0][-2:]}.o"
            result = subprocess.run(
                ["c++", "-std=c++17", "-Wall", "-Wextra", "-Wpedantic",
                 *["-D" + d for d in macros.split()], "-I", str(CPP / "inc"),
                 "-I", str(YOLOV5_INC), "-c", str(source), "-o", str(obj)],
                capture_output=True, text=True, timeout=COMPILE_TIMEOUT)
            self.assertEqual(result.returncode, 0, macros + ": " + result.stderr)

    # ----------------------------------------------------------- behaviour ----

    def test_behaviour_harness_passes(self):
        result = subprocess.run([str(self.checks)], capture_output=True, text=True,
                                timeout=RUN_TIMEOUT)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("ALL CHECKS PASSED", result.stdout)

    def test_component_goldens_pass(self):
        # IoU goldens come from the baseline's cython_bbox; assignment goldens
        # from lap.lapjv(extend_cost=True, cost_limit=...). Bit-exact equality
        # is the point, so any mismatch fails the build's parity story.
        result = subprocess.run([str(self.components)], capture_output=True, text=True,
                                timeout=RUN_TIMEOUT)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("ALL COMPONENT CHECKS PASSED", result.stdout)

    def test_replay_cli_reports_and_writes_records(self):
        stream = person_stream(3, 40)
        detections = self.tmp / "three-persons.btcpp"
        records = self.tmp / "three-persons.tsv"
        write_replay(detections, stream)
        result = subprocess.run(
            [str(self.replay_cli), "replay", "--detections", str(detections),
             "--records", str(records)],
            capture_output=True, text=True, timeout=RUN_TIMEOUT)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("replay: 40 frames", result.stdout)

        lines = records.read_text().splitlines()
        self.assertEqual(lines[0] + "\n", RECORDS_HEADER)
        rows = [line.split("\t") for line in lines[1:]]
        self.assertTrue(rows, "records must contain track rows")
        ids = set()
        for row in rows:
            self.assertEqual(len(row), 8)
            frame, lane, track_id = int(row[0]), int(row[1]), int(row[2])
            self.assertIn(frame, range(1, 41))
            self.assertEqual(lane, 0)
            ids.add(track_id)
            values = [float(v) for v in row[3:]]
            self.assertTrue(all(v == v and -1e9 < v < 1e9 for v in values), row)
        # Three stable persons with churn from occlusion/low-score windows stay
        # a handful of IDs, and no ID may ever be reused after removal.
        self.assertLessEqual(len(ids), 12, f"unexpected ID churn: {sorted(ids)}")
        self.assertGreaterEqual(len(ids), 3)
        # Frame 1: high-score detections echo their boxes exactly (the Kalman
        # mean initializes from the measurement; integer coordinates round-trip
        # through float32 and %.17g without loss).
        frame_one = [row for row in rows if int(row[0]) == 1]
        self.assertEqual(len(frame_one), 3)
        echoed = sorted(
            (round(float(row[3])), round(float(row[4])), round(float(row[5])),
             round(float(row[6])), int(row[2]))
            for row in frame_one)
        self.assertEqual(echoed, [(100, 200, 160, 350, 1), (260, 290, 320, 440, 2),
                                  (420, 380, 480, 530, 3)])

    def test_replay_cli_rejects_broken_inputs(self):
        good = self.tmp / "good.btcpp"
        write_replay(good, [[(10, 10, 60, 160, 0.9)]])
        payload = good.read_bytes()
        cases = {
            "empty": b"",
            "bad magic": b"NOTMAGIC" + payload[8:],
            "truncated header": payload[:12],
            "truncated frame": payload[:-4],
            "count past end": payload[:28],  # declares 1 det, zero bytes follow
        }
        for name, blob in cases.items():
            broken = self.tmp / f"broken-{name.replace(' ', '-')}.btcpp"
            broken.write_bytes(blob)
            result = subprocess.run(
                [str(self.replay_cli), "replay", "--detections", str(broken)],
                capture_output=True, text=True, timeout=RUN_TIMEOUT)
            self.assertNotEqual(result.returncode, 0, name)
            self.assertTrue(result.stderr.strip(), name)
        # Missing --detections is a usage error.
        result = subprocess.run([str(self.replay_cli), "replay"],
                                capture_output=True, text=True, timeout=RUN_TIMEOUT)
        self.assertEqual(result.returncode, 2)
        # The video mode is a board activity; a host binary must refuse it.
        result = subprocess.run(
            [str(self.replay_cli), "video", "--input", "x.mp4", "--model-path", "y.bin"],
            capture_output=True, text=True, timeout=RUN_TIMEOUT)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("board", result.stderr.lower())

    def test_cli_numeric_parsing_contract(self):
        # The parser contract: finite-only (NaN/Inf rejected before
        # any cast), whole integers (no silent 5.5 -> 5 truncation), and
        # numeric_limits<float>::min() is NOT a lower bound (threshold 0 is a
        # valid value).
        stream = person_stream(1, 6)
        good = self.tmp / "cli-num.btcpp"
        write_replay(good, stream)
        cases_ok = [
            ["replay", "--detections", str(good), "--track-thresh", "0"],
            ["replay", "--detections", str(good), "--track-buffer", "0"],
            ["replay", "--detections", str(good), "--threads", "4"],
        ]
        for case in cases_ok:
            result = subprocess.run([str(self.replay_cli), *case],
                                    capture_output=True, text=True, timeout=RUN_TIMEOUT)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        cases_bad = [
            ["replay", "--detections", str(good), "--track-buffer", "5.5"],   # fractional int
            ["replay", "--detections", str(good), "--threads", "2.5"],
            ["replay", "--detections", str(good), "--track-thresh", "nan"],
            ["replay", "--detections", str(good), "--track-thresh", "inf"],
            ["replay", "--detections", str(good), "--track-thresh", "1.5"],   # semantic [0,1]
            ["replay", "--detections", str(good), "--track-thresh", "oops"],
        ]
        for case in cases_bad:
            result = subprocess.run([str(self.replay_cli), *case],
                                    capture_output=True, text=True, timeout=RUN_TIMEOUT)
            self.assertNotEqual(result.returncode, 0, case)
            self.assertTrue(result.stderr.strip(), case)
        # --help prints usage and exits 0 without requiring any input.
        result = subprocess.run([str(self.replay_cli), "--help"],
                                capture_output=True, text=True, timeout=RUN_TIMEOUT)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("usage: bytetrack", result.stdout)

    def test_cli_rejects_documented_ranges(self):
        stream = person_stream(1, 6)
        good = self.tmp / "cli-range.btcpp"
        write_replay(good, stream)
        cases = [
            ["replay", "--detections", str(good), "--max-frames", "-1"],
            ["replay", "--detections", str(good), "--score-thres", "1.5"],
            ["replay", "--detections", str(good), "--nms-thres", "-0.1"],
            ["replay", "--detections", str(good), "--priority", "256"],
            ["replay", "--detections", str(good), "--priority", "-1"],
            ["replay", "--detections", str(good), "--bpu-cores", "4"],
        ]
        for case in cases:
            result = subprocess.run([str(self.replay_cli), *case],
                                    capture_output=True, text=True, timeout=RUN_TIMEOUT)
            self.assertNotEqual(result.returncode, 0, case)
            self.assertTrue(result.stderr.strip(), case)

    def test_cli_rejects_output_paths_that_name_an_input(self):
        # Writers truncate on open: a requested output colliding with an
        # input (or the other writer's file) must be refused BEFORE anything
        # is opened, and the input's content must survive the attempt. The
        # same-path, aliased-path (symlink) and not-yet-existing-path cases
        # all count as collisions.
        import os
        stream = person_stream(1, 6)
        good = self.tmp / "conflict.btcpp"
        write_replay(good, stream)
        original = good.read_bytes()
        alias = self.tmp / "conflict-alias.btcpp"
        os.symlink(good, alias)
        cases = [
            ["replay", "--detections", str(good), "--records", str(good)],
            ["replay", "--detections", str(alias), "--records", str(good)],
            ["replay", "--detections", str(good), "--records",
             str(self.tmp / "pending" / ".." / "conflict.btcpp")],
            # video mode: host binary refuses video before path checks, so
            # the replay cases above carry the collision contract.
        ]
        for case in cases:
            result = subprocess.run([str(self.replay_cli), *case],
                                    capture_output=True, text=True, timeout=RUN_TIMEOUT)
            self.assertNotEqual(result.returncode, 0, case)
            self.assertIn("overwrite an input", result.stderr, case)
        self.assertEqual(good.read_bytes(), original)

    def test_bench_multi_lane_aborts_on_invalid_detections(self):
        # A lane worker throwing must abort every waiter, join all workers
        # and exit non-zero — never wedge an == waiter on a counter another
        # lane can still increment. A NaN score mid-stream (inside the timed
        # frames, after warmup) is rejected by ByteTracker::update. The
        # subprocess timeout is the regression guard: a deadlock fails the
        # test instead of hanging it.
        frames = person_stream(3, 40)
        frames[12][0] = (10, 10, 60, 160, float("nan"))
        detections = self.tmp / "bench-invalid.btcpp"
        write_replay(detections, frames)
        for lanes in (1, 2, 4):
            result = subprocess.run(
                [str(self.bench), "--detections", str(detections),
                 "--warmup", "5", "--frames", "20", "--repeats", "1",
                 "--threads", "2", "--lanes", str(lanes)],
                capture_output=True, text=True, timeout=60)
            self.assertNotEqual(result.returncode, 0, f"lanes={lanes}")
            self.assertIn("error: tracker raised", result.stderr, f"lanes={lanes}")

    def test_bench_harness_smoke(self):
        stream = person_stream(5, 40)
        detections = self.tmp / "bench.btcpp"
        write_replay(detections, stream)
        csv = self.tmp / "bench.csv"
        js = self.tmp / "bench.json"
        result = subprocess.run(
            [str(self.bench), "--detections", str(detections),
             "--warmup", "2", "--frames", "5", "--repeats", "1",
             "--threads", "1", "--lanes", "1",
             "--csv", str(csv), "--json", str(js)],
            capture_output=True, text=True, timeout=RUN_TIMEOUT)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("repeat 0 threads 1 lanes 1", result.stdout)
        lines = csv.read_text().splitlines()
        self.assertEqual(lines[0], "threads,lanes,repeat,lane,frame,n_in,n_out,wall_ns,cpu_ns")
        self.assertEqual(len(lines), 1 + 5)  # one row per timed frame
        for line in lines[1:]:
            fields = line.split(",")
            self.assertEqual(len(fields), 9)
            self.assertTrue(int(fields[7]) > 0, line)  # wall_ns strictly positive
        summary = json.loads(js.read_text())
        self.assertEqual(summary["warmup"], 2)
        self.assertEqual(summary["frames"], 5)
        self.assertEqual(summary["img"], [1080, 1920])
        self.assertEqual(summary["n_frames_file"], 40)

    def test_s_adapter_contract_of_the_committed_yolov5_file(self):
        # ByteTrack's board video path composes this exact translation unit.
        # Two properties the four-target delivery depends on (each verified
        # behaviorally on the boards): the S adapter's target whitelist
        # covers all three S targets behind the build-identity gate, and one
        # UCP task is released per infer call instead of leaking every task
        # but the last until model release.
        src = YOLOV5_DETECT.read_text()
        gate = ('require_gate(check_target_matches_build(config.target, '
                'YOLOV5_TARGET_NAME));')
        self.assertIn(gate, src)
        whitelist = 'if (config.target != "s100" && config.target != "s100p" && config.target != "s600")'
        self.assertIn(whitelist, src)
        self.assertIn('S adapter supports s100, s100p and s600 YOLOv5 assets only', src)
        release = ("const int released = hbUCPReleaseTask(guard.task);")
        self.assertIn(release, src)
        self.assertIn("guard.task = nullptr;", src)
        # The per-call release must come after the output reads and null the
        # handle before checking the result (a failed release must not be
        # double-released by the resource guard's destructor).
        infer = src.index("RawResult infer(const Prepared& prepared)")
        self.assertIn(release, src[infer:])
        self.assertLess(src.index(release), src.index("return raw;", src.index(release)))


if __name__ == "__main__":
    unittest.main()
