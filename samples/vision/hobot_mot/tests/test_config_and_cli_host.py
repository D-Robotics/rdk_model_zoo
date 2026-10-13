"""Host-side behavior tests for the HobotMot sample.

These tests cover everything that must work without the ARM64 native
library: config validation, detection-JSONL contract, CLI parsing, result
formatting and the native-library discovery error path. Tests that need
libhobot_mot_capi.so live in test_track_native.py and run on the boards.
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
SAMPLE_ROOT = TESTS_DIR.parent
RUNTIME_PY = SAMPLE_ROOT / "runtime" / "python"
sys.path.insert(0, str(RUNTIME_PY))

from cli import (DEFAULT_INPUT, build_parser, load_frame_records,  # noqa: E402
                  render_frames, save_results)
from main import main as main_entry  # noqa: E402
from native import MotNativeError, find_native_library  # noqa: E402
from track import (DEFAULT_CONFIG, Detection, HobotMotTracker,  # noqa: E402
                   MotConfigError, MotInputError, _resolve_dimension,
                   validate_config, validate_config_file)

CONFIG_DIR = SAMPLE_ROOT / "model" / "configs"
TEST_DATA = SAMPLE_ROOT / "test_data" / "detections.jsonl"


class SuppliedConfigsTest(unittest.TestCase):
    def test_supplied_configs_validate(self) -> None:
        for name in ("iou2.json", "iou2_kalman.json"):
            with self.subTest(config=name):
                config = validate_config_file(CONFIG_DIR / name)
                self.assertEqual(config["tracker_type"], "IOU_2.0")

    def test_kalman_config_is_real_bool(self) -> None:
        # The upstream reader ignores integer 1 (isBool() only); the shipped
        # kalman config must use a JSON bool to actually enable the filter.
        config = json.loads((CONFIG_DIR / "iou2_kalman.json").read_text())
        self.assertIsInstance(config["use_kalman_filter"], bool)
        self.assertTrue(config["use_kalman_filter"])


class ConfigValidationTest(unittest.TestCase):
    def base(self) -> dict:
        return json.loads((CONFIG_DIR / "iou2.json").read_text())

    def test_rejects_wrong_tracker_type(self) -> None:
        for value in ("IOU", "ReID", "bogus", None, 1):
            with self.subTest(value=value):
                config = self.base()
                config["tracker_type"] = value
                with self.assertRaises(MotConfigError):
                    validate_config(config)

    def test_rejects_missing_tracker_type(self) -> None:
        config = self.base()
        del config["tracker_type"]
        with self.assertRaises(MotConfigError):
            validate_config(config)

    def test_rejects_integer_for_bool_field(self) -> None:
        # Upstream SET_MOT_METHOD_PARAM reads isBool(); integer 1 keeps the
        # default silently, so it must be rejected loudly.
        config = self.base()
        config["use_kalman_filter"] = 1
        with self.assertRaises(MotConfigError):
            validate_config(config)

    def test_accepts_bool_for_bool_field(self) -> None:
        config = self.base()
        config["use_kalman_filter"] = False
        validate_config(config)

    def test_rejects_bool_for_int_field(self) -> None:
        config = self.base()
        config["missing_time_thres"] = True
        with self.assertRaises(MotConfigError):
            validate_config(config)

    def test_rejects_out_of_int32_integers(self) -> None:
        config = self.base()
        config["missing_time_thres"] = 2 ** 40
        with self.assertRaises(MotConfigError):
            validate_config(config)

    def test_rejects_out_of_uint32_integers(self) -> None:
        config = self.base()
        config["vanish_frame_count"] = 2 ** 40
        with self.assertRaises(MotConfigError):
            validate_config(config)

    def test_rejects_bad_match_type(self) -> None:
        config = self.base()
        config["match_type"] = "magic"
        with self.assertRaises(MotConfigError):
            validate_config(config)

    def test_rejects_out_of_range_doubles(self) -> None:
        config = self.base()
        config["iou_thres"] = 1.5
        with self.assertRaises(MotConfigError):
            validate_config(config)

    def test_ignores_unknown_keys(self) -> None:
        # Upstream ignores fields it does not read; validation must too.
        config = self.base()
        config["some_future_field"] = {"nested": True}
        validate_config(config)

    def test_missing_file_is_config_error(self) -> None:
        with self.assertRaises(MotConfigError):
            validate_config_file("/nonexistent/hobot-mot.json")

    def test_invalid_json_is_config_error(self) -> None:
        bad = TESTS_DIR / "_bad_config.json"
        bad.write_text("{not json", encoding="utf-8")
        try:
            with self.assertRaises(MotConfigError):
                validate_config_file(bad)
        finally:
            bad.unlink()


class DetectionSequenceTest(unittest.TestCase):
    def test_supplied_sequence_loads(self) -> None:
        records = load_frame_records(TEST_DATA)
        self.assertEqual(len(records), 72)
        classes = {det.class_id for record in records
                   for det in record.detections}
        self.assertEqual(classes, {0, 1})

    def test_sequence_covers_documented_scenarios(self) -> None:
        records = load_frame_records(TEST_DATA)
        by_index = {record.frame_index: record for record in records}
        self.assertEqual(by_index[10].detections, [])  # empty frame ages all
        # person B temporarily missing on frame 6, back on frame 7
        self.assertEqual([d.class_id for d in by_index[6].detections], [0, 1])
        self.assertEqual([d.class_id for d in by_index[7].detections],
                         [0, 0, 1])
        # low-score decoy filtered by min_score on frames 12-14
        scores = [d.score for d in by_index[13].detections]
        self.assertIn(0.3, scores)
        # car C (class 1) leaves after frame 19
        self.assertTrue(any(d.class_id == 1 for d in by_index[19].detections))
        self.assertFalse(any(d.class_id == 1 for d in by_index[20].detections))

    def test_rejects_non_increasing_timestamps(self) -> None:
        bad = TESTS_DIR / "_bad_seq.jsonl"
        bad.write_text(
            json.dumps({"frame_index": 0, "timestamp_ms": 100,
                        "detections": []}) + "\n" +
            json.dumps({"frame_index": 1, "timestamp_ms": 100,
                        "detections": []}) + "\n",
            encoding="utf-8")
        try:
            with self.assertRaises(ValueError):
                load_frame_records(bad)
        finally:
            bad.unlink()

    def test_rejects_missing_required_keys(self) -> None:
        bad = TESTS_DIR / "_bad_seq.jsonl"
        bad.write_text(json.dumps({"frame_index": 0}) + "\n", encoding="utf-8")
        try:
            with self.assertRaises(ValueError):
                load_frame_records(bad)
        finally:
            bad.unlink()

    def test_rejects_bad_detection_rows(self) -> None:
        bad = TESTS_DIR / "_bad_seq.jsonl"
        record = {"frame_index": 0, "timestamp_ms": 0,
                  "detections": [{"bbox": [1, 2, 3], "score": 0.9,
                                  "class_id": 0}]}
        bad.write_text(json.dumps(record) + "\n", encoding="utf-8")
        try:
            with self.assertRaises(ValueError):
                load_frame_records(bad)
        finally:
            bad.unlink()

    def test_rejects_out_of_range_score(self) -> None:
        bad = TESTS_DIR / "_bad_seq.jsonl"
        record = {"frame_index": 0, "timestamp_ms": 0,
                  "detections": [{"bbox": [1, 2, 3, 4], "score": 1.5,
                                  "class_id": 0}]}
        bad.write_text(json.dumps(record) + "\n", encoding="utf-8")
        try:
            with self.assertRaises(ValueError):
                load_frame_records(bad)
        finally:
            bad.unlink()


class CliContractTest(unittest.TestCase):
    def test_defaults_match_documented_values(self) -> None:
        args = build_parser().parse_args([])
        self.assertEqual(args.frame_width, 960)
        self.assertEqual(args.frame_height, 544)
        self.assertEqual(args.config, str(DEFAULT_CONFIG))
        self.assertEqual(args.input, str(DEFAULT_INPUT))
        self.assertIsNone(args.output)

    def test_help_works_without_native_library(self) -> None:
        import contextlib
        import io
        parser = build_parser()
        with self.assertRaises(SystemExit) as ctx:
            with contextlib.redirect_stdout(io.StringIO()):
                parser.parse_args(["--help"])
        self.assertEqual(ctx.exception.code, 0)

    def test_frame_size_requires_integers(self) -> None:
        with self.assertRaises(SystemExit):
            build_parser().parse_args(["--frame-width", "wide"])


class MainEntryTest(unittest.TestCase):
    """The visible main: same contract as the C++ binary's main()."""

    def test_help_exits_zero(self) -> None:
        import contextlib
        import io
        with self.assertRaises(SystemExit) as ctx:
            with contextlib.redirect_stdout(io.StringIO()):
                main_entry(["--help"])
        self.assertEqual(ctx.exception.code, 0)

    def test_run_without_native_library_fails_cleanly(self) -> None:
        import contextlib
        import io
        build_dir = SAMPLE_ROOT / "runtime/cpp/build"
        if any(build_dir.glob("**/libhobot_mot_capi.so")):
            self.skipTest("native library present (board checkout)")
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            code = main_entry([])
        self.assertEqual(code, 2)
        self.assertIn("libhobot_mot_capi.so", stderr.getvalue())
        self.assertIn("Error:", stderr.getvalue())

    def test_invalid_input_file_fails_cleanly(self) -> None:
        import contextlib
        import io
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            code = main_entry(["--input", "/nonexistent/detections.jsonl"])
        self.assertEqual(code, 2)
        self.assertIn("Error:", stderr.getvalue())


class DimensionValidationTest(unittest.TestCase):
    """Frame dimensions must be exact integers; no int() coercion."""

    def test_resolve_dimension_semantics(self) -> None:
        # None and 0 mean "unspecified" (0 mirrors the C++ marker).
        self.assertEqual(_resolve_dimension(None, 960, "frame_width"), 960)
        self.assertEqual(_resolve_dimension(0, 960, "frame_width"), 960)
        self.assertEqual(_resolve_dimension(640, 960, "frame_width"), 640)
        for bad in (True, 1.5, "100", -5, 2 ** 31):
            with self.subTest(value=bad):
                with self.assertRaises(MotInputError):
                    _resolve_dimension(bad, 960, "frame_width")

    def test_init_rejects_non_integer_dimensions(self) -> None:
        # Checked before the config is opened or the native library is
        # loaded, so this is testable on a host without a board build.
        for bad in (True, 1.5, "100"):
            with self.subTest(value=bad):
                with self.assertRaises(MotConfigError):
                    HobotMotTracker(frame_width=bad)

    def test_init_rejects_out_of_range_dimensions(self) -> None:
        for bad in (0, -5, 2 ** 31):
            with self.subTest(value=bad):
                with self.assertRaises(MotConfigError):
                    HobotMotTracker(frame_height=bad)


class NativeDiscoveryTest(unittest.TestCase):
    def test_missing_library_raises_actionable_error(self) -> None:
        # On hosts without the board build, discovery must fail with build
        # instructions instead of a fake success.
        build_dir = SAMPLE_ROOT / "runtime/cpp/build"
        had_lib = any(build_dir.glob("**/libhobot_mot_capi.so"))
        if had_lib:
            self.skipTest("native library present (board checkout)")
        with self.assertRaises(MotNativeError) as ctx:
            find_native_library()
        self.assertIn("cmake", str(ctx.exception))


class ResultRenderingTest(unittest.TestCase):
    def test_render_and_save_roundtrip(self) -> None:
        from track import DisappearedId, FrameTracks, TrackResult
        result = FrameTracks(
            timestamp_ms=1040,
            tracks=[TrackResult((100, 100, 160, 260), 0.949999988079071, 0,
                                1, "valid")],
            disappeared=[DisappearedId(1, 3)])
        record_path = TESTS_DIR / "_render.json"
        save_results(record_path, [], [])
        saved = json.loads(record_path.read_text())
        self.assertEqual(saved, [])
        record_path.unlink()
        # save_results pairs records with results; one record, one result.
        from cli import FrameRecord
        records = [FrameRecord(0, 1040, 640, 480, [])]
        save_results(record_path, records, [result])
        saved = json.loads(record_path.read_text())
        self.assertEqual(saved[0]["tracks"][0]["track_id"], 1)
        record_path.unlink()


if __name__ == "__main__":
    unittest.main()
