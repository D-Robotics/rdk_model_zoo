"""Native tracker tests; they run wherever libhobot_mot_capi.so is built.

On a host without the board build every test skips (see
test_config_and_cli_host.py for the no-native coverage). On an RDK board the
suite exercises the real upstream IOU 2.0 core through the sample's C ABI:
identity assignment and namespaces, input-order alignment, aging on empty and
missing frames, reset semantics, instance isolation and the validation
boundaries that must reject bad input before any native call.
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
SAMPLE_ROOT = TESTS_DIR.parent
RUNTIME_PY = SAMPLE_ROOT / "runtime" / "python"
sys.path.insert(0, str(RUNTIME_PY))

from cli import DEFAULT_INPUT  # noqa: E402
from native import find_native_library  # noqa: E402
from track import (DEFAULT_CONFIG, Detection, HobotMotTracker,  # noqa: E402
                   MotConfigError, MotInputError)

try:
    NATIVE_LIBRARY = find_native_library()
except Exception:  # noqa: BLE001 - any discovery failure means "skip"
    NATIVE_LIBRARY = None


def det(x1, y1, x2, y2, score=0.95, class_id=0) -> Detection:
    return Detection((x1, y1, x2, y2), score, class_id)


@unittest.skipUnless(NATIVE_LIBRARY, "libhobot_mot_capi.so not built (board)")
class IdentitySemanticsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tracker = HobotMotTracker()

    def tearDown(self) -> None:
        self.tracker.close()

    def test_first_frame_assigns_distinct_ids_per_class(self) -> None:
        result = self.tracker.predict(
            [det(100, 100, 160, 260), det(400, 300, 460, 460),
             det(480, 200, 600, 300, class_id=1)], 1000)
        ids = [track.track_id for track in result.tracks]
        # class-agnostic upstream, one instance per class: each namespace
        # numbers its own identities from 1.
        self.assertEqual(ids, [1, 2, 1])
        self.assertEqual([track.class_id for track in result.tracks],
                         [0, 0, 1])

    def test_rows_stay_aligned_with_input_order(self) -> None:
        detections = [det(480, 200, 600, 300, class_id=1),
                      det(100, 100, 160, 260),
                      det(400, 300, 460, 460)]
        result = self.tracker.predict(detections, 1000)
        self.assertEqual([track.class_id for track in result.tracks],
                         [1, 0, 0])

    def test_persistent_objects_keep_their_ids(self) -> None:
        first = self.tracker.predict(
            [det(100, 100, 160, 260), det(400, 300, 460, 460)], 1000)
        for frame in range(1, 10):
            result = self.tracker.predict(
                [det(100 + 4 * frame, 100, 160 + 4 * frame, 260),
                 det(400, 300, 460, 460)], 1000 + 40 * frame)
            self.assertEqual([t.track_id for t in result.tracks],
                             [t.track_id for t in first.tracks])
            self.assertTrue(all(t.state == "valid" for t in result.tracks))

    def test_unrelated_classes_do_not_share_identities(self) -> None:
        # Same geometry, different class: two namespaces, both start at 1.
        result = self.tracker.predict(
            [det(100, 100, 160, 260), det(100, 100, 160, 260, class_id=7)],
            1000)
        self.assertEqual([t.track_id for t in result.tracks], [1, 1])

    def test_empty_frame_then_reappearance_keeps_id(self) -> None:
        first = self.tracker.predict([det(100, 100, 160, 260)], 1000)
        original = first.tracks[0].track_id
        empty = self.tracker.predict([], 1040)
        self.assertEqual(empty.tracks, [])
        back = self.tracker.predict([det(112, 100, 172, 260)], 1080)
        self.assertEqual(back.tracks[0].track_id, original)

    def test_short_gap_keeps_id(self) -> None:
        # A detection missing for a few frames (within vanish_frame_count)
        # must not re-identify the object when it comes back.
        first = self.tracker.predict([det(100, 100, 160, 260)], 1000)
        original = first.tracks[0].track_id
        for frame in range(1, 6):
            self.tracker.predict([], 1000 + 40 * frame)
        back = self.tracker.predict([det(140, 100, 200, 260)], 1240)
        self.assertEqual(back.tracks[0].track_id, original)

    def test_low_score_rows_are_not_tracked(self) -> None:
        # Below min_score upstream keeps the row for alignment but assigns
        # no identity: id -1, state invalid (board-verified behavior).
        result = self.tracker.predict(
            [det(100, 100, 160, 260), det(300, 50, 340, 150, score=0.3)],
            1000)
        kept, dropped = result.tracks
        self.assertEqual(kept.state, "valid")
        self.assertGreaterEqual(kept.track_id, 1)
        self.assertEqual(dropped.track_id, -1)
        self.assertEqual(dropped.state, "invalid")


@unittest.skipUnless(NATIVE_LIBRARY, "libhobot_mot_capi.so not built (board)")
class LifecycleTest(unittest.TestCase):
    def test_reset_restarts_identities_and_timestamp_gate(self) -> None:
        tracker = HobotMotTracker()
        try:
            first = tracker.predict([det(100, 100, 160, 260)], 5000)
            tracker.predict([det(104, 100, 164, 260)], 5040)
            tracker.reset()
            # After reset the tracker behaves like a new one: timestamps may
            # restart and identities number from 1 again.
            restarted = tracker.predict([det(100, 100, 160, 260)], 1000)
            self.assertEqual(restarted.tracks[0].track_id, 1)
            self.assertEqual(first.tracks[0].track_id, 1)
        finally:
            tracker.close()

    def test_instances_are_isolated(self) -> None:
        sequence = [
            ([det(100, 100, 160, 260), det(400, 300, 460, 460)], 1000),
            ([det(104, 100, 164, 260), det(400, 300, 460, 460)], 1040),
            ([det(108, 100, 168, 260)], 1080),
        ]
        outputs = []
        for _ in range(2):
            tracker = HobotMotTracker()
            try:
                outputs.append([tracker.predict(dets, ts)
                                for dets, ts in sequence])
            finally:
                tracker.close()
        # Advancing one instance must not leak into the other.
        def ids_per_frame(frames):
            return [[t.track_id for t in frame.tracks] for frame in frames]

        self.assertEqual(ids_per_frame(outputs[0]), ids_per_frame(outputs[1]))

    def test_config_mutation_after_construction_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / "iou2.json"
            config.write_text(json.dumps(json.loads(
                Path(DEFAULT_CONFIG).read_text())), encoding="utf-8")
            tracker = HobotMotTracker(config_path=config)
            try:
                tracker.predict([det(100, 100, 160, 260)], 1000)
                # A second class forces a new upstream instance; an edited
                # config must be rejected, not silently run with new values.
                config.write_text(json.dumps({
                    **json.loads(config.read_text()), "iou_thres": 0.9}),
                    encoding="utf-8")
                with self.assertRaises(MotConfigError):
                    tracker.predict([det(100, 100, 160, 260, class_id=1)],
                                    1040)
            finally:
                tracker.close()

    def test_kalman_config_runs(self) -> None:
        tracker = HobotMotTracker(
            config_path=SAMPLE_ROOT / "model" / "configs" / "iou2_kalman.json")
        try:
            result = tracker.predict([det(100, 100, 160, 260)], 1000)
            self.assertGreaterEqual(result.tracks[0].track_id, 1)
        finally:
            tracker.close()


@unittest.skipUnless(NATIVE_LIBRARY, "libhobot_mot_capi.so not built (board)")
class InputBoundaryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tracker = HobotMotTracker()

    def tearDown(self) -> None:
        self.tracker.close()

    def test_timestamp_must_strictly_increase(self) -> None:
        self.tracker.predict([det(100, 100, 160, 260)], 1000)
        with self.assertRaises(MotInputError):
            self.tracker.predict([det(104, 100, 164, 260)], 1000)
        with self.assertRaises(MotInputError):
            self.tracker.predict([det(104, 100, 164, 260)], 960)

    def test_out_of_range_integers_rejected_before_native(self) -> None:
        # Values ctypes could not marshal without wrapping must be rejected
        # by the Python gate, never truncated inside the C ABI.
        with self.assertRaises(MotInputError):
            self.tracker.predict([det(100, 100, 160, 260)], 2 ** 63)
        with self.assertRaises(MotInputError):
            self.tracker.predict(
                [Detection((0, 0, 10, 10), 0.9, True)], 1000)
        with self.assertRaises(MotInputError):
            self.tracker.predict(
                [Detection((-2 ** 40, 0, 10, 10), 0.9, 0)], 1000)

    def test_degenerate_boxes_rejected(self) -> None:
        with self.assertRaises(MotInputError):
            self.tracker.predict([det(100, 100, 100, 260)], 1000)
        with self.assertRaises(MotInputError):
            self.tracker.predict([det(200, 100, 100, 260)], 1000)

    def test_score_bounds(self) -> None:
        result = self.tracker.predict(
            [det(100, 100, 160, 260, score=1.0)], 1000)
        self.assertEqual(result.tracks[0].state, "valid")
        with self.assertRaises(MotInputError):
            self.tracker.predict([det(100, 100, 160, 260, score=1.5)], 1040)

    def test_infer_revalidates_staged_input(self) -> None:
        prepared = self.tracker.preprocess([det(100, 100, 160, 260)], 1000)
        self.tracker.infer(prepared)
        # A staged caller cannot replay the same timestamp through infer.
        with self.assertRaises(MotInputError):
            self.tracker.infer(prepared)

    def test_preprocess_rejects_coercible_dimensions(self) -> None:
        # int() coercion would silently turn True/1.5/"100" into a frame
        # size; they must be rejected as invalid dimensions instead.
        for bad in (True, 1.5, "100", -5):
            with self.subTest(value=bad):
                with self.assertRaises(MotInputError):
                    self.tracker.preprocess([], 1000, bad, None)
                with self.assertRaises(MotInputError):
                    self.tracker.preprocess([], 1000, None, bad)

    def test_preprocess_zero_dimension_means_default(self) -> None:
        # 0 mirrors the C++ "unspecified" marker: the instance default runs.
        prepared = self.tracker.preprocess([], 1000, 0, 0)
        self.assertEqual(prepared.frame_width, 960)
        self.assertEqual(prepared.frame_height, 544)


@unittest.skipUnless(NATIVE_LIBRARY, "libhobot_mot_capi.so not built (board)")
class SuppliedSequenceReplayTest(unittest.TestCase):
    def test_full_replay_matches_upstream_expectations(self) -> None:
        from cli import load_frame_records
        records = load_frame_records(DEFAULT_INPUT)
        tracker = HobotMotTracker()
        try:
            results = [tracker.predict(record.detections, record.timestamp_ms,
                                       record.image_width or None,
                                       record.image_height or None)
                       for record in records]
        finally:
            tracker.close()
        self.assertEqual(len(results), len(records))

        # Class 0 holds exactly two valid identities the whole sequence:
        # walker A (id 1) and static B (id 2). The low-score decoy near
        # x=300 never becomes valid (it cannot be separated from A by
        # position alone once A walks past x=300, so the identity set is
        # the assertion, not per-position buckets).
        valid_class0 = {t.track_id for result in results
                        for t in result.tracks
                        if t.class_id == 0 and t.state == "valid"}
        self.assertEqual(valid_class0, {1, 2})

        # The car (class 1) is last seen on frame 19 and must be reported
        # removed exactly once, after the vanish window (50 x 40 ms).
        removals = [gone for result in results for gone in result.disappeared]
        car_removals = [gone for gone in removals if gone.class_id == 1]
        self.assertEqual(len(car_removals), 1)
        removal_frame = next(
            record.frame_index for record, result in zip(records, results)
            if any(g.class_id == 1 for g in result.disappeared))
        self.assertGreaterEqual(removal_frame, 60)
        self.assertLessEqual(removal_frame, 71)
        # And it never comes back.
        post = [t for record, result in zip(records, results)
                if record.frame_index > removal_frame for t in result.tracks
                if t.class_id == 1]
        self.assertEqual(post, [])


if __name__ == "__main__":
    unittest.main()
