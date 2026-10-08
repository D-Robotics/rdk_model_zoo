"""The readable stage surface (preprocess/infer/postprocess) on ByteTrackTask.

These tests pin the accepted readable-runtime contract for the tracking task:
the new stage names carry the real implementation, the established
``pre_process``/``forward``/``post_process`` names delegate to them (no second
implementation), ``predict`` equals the explicit three-step chain, and the
tracker is updated exactly once per frame — including frames whose detections
are all filtered out. Host fixtures only; no board SDK is loaded and no board
inference is claimed.
"""

from unittest import mock
import unittest
from pathlib import Path
import numpy as np

from samples.vision.bytetrack.runtime.python.tracking import ByteTrackTask
from test_tracking import FakeDetector, FakeTracker


def _task():
    return ByteTrackTask(FakeDetector(), tracker_factory=FakeTracker)


class ByteTrackReadableStageTests(unittest.TestCase):
    def test_new_stage_names_and_predict_equality(self):
        task = _task()
        image = np.zeros((17, 31, 3), np.uint8)
        prepared = task.preprocess(image)
        staged = task.postprocess(task.infer(prepared.tensors), prepared.context)
        composed = task.predict(image)
        self.assertEqual(len(staged), 1)
        self.assertEqual(task.frame_index, 2)
        for left, right in zip(staged, composed):
            self.assertEqual(left.track_id, right.track_id)
            self.assertEqual(left.tlbr, right.tlbr)
            self.assertEqual(left.score, right.score)

    def test_legacy_stage_names_delegate_to_the_new_methods(self):
        task = _task()
        image = np.zeros((17, 31, 3), np.uint8)
        with mock.patch.object(task, 'preprocess', wraps=task.preprocess) as pre, \
                mock.patch.object(task, 'infer', wraps=task.infer) as infer, \
                mock.patch.object(task, 'postprocess', wraps=task.postprocess) as post:
            prepared = task.pre_process(image)
            task.post_process(task.forward(prepared.tensors), prepared.context)
        pre.assert_called_once()
        infer.assert_called_once()
        post.assert_called_once()

    def test_predict_updates_the_tracker_exactly_once_per_frame(self):
        task = _task()
        for _ in range(3):
            before = task.frame_index
            task.predict(np.zeros((17, 31, 3), np.uint8))
            self.assertEqual(task.frame_index, before + 1)
        self.assertEqual(len(task.tracker.calls), 3)

    def test_empty_detections_still_update_tracker_and_reset_keeps_ids(self):
        detector = FakeDetector()
        task = ByteTrackTask(detector, tracker_factory=FakeTracker)
        # Non-person classes filter every detection out of the Kalman input.
        detector.result.class_ids.fill(3)
        self.assertEqual(task.predict(np.zeros((31, 17, 3), np.uint8)), ())
        self.assertEqual(task.tracker.calls[-1][0].shape, (0, 5))
        self.assertEqual(task.frame_index, 1)
        task.reset()
        self.assertEqual(task.frame_index, 0)
        detector.result.class_ids[:] = np.array([0, 1], np.int32)
        self.assertEqual(task.predict(np.zeros((31, 17, 3), np.uint8))[0].track_id, 7)


if __name__ == '__main__':
    unittest.main()


class SimplifiedRuntimeTests(unittest.TestCase):
    """2026-10-08 runtime simplification boundary.

    Selection/catalog duties live in ``cli.py``; ``tracking.py`` owns the
    named model class whose ``from_model`` constructs the detector and loads
    the runtime; the per-sample ``model_binding``/``model_runner`` forwarding
    modules are gone. ``tracker_backend/`` stays: the substantial BYTETracker
    algorithm (Kalman filter, matching, tracker state).
    """

    def test_from_model_owns_detector_construction_and_streams(self):
        from samples.vision.bytetrack.runtime.python.cli import resolve_selection
        from samples.vision.bytetrack.runtime.python.tracking import ByteTrackTask
        from samples.vision.yolov5.tests.test_yolov5 import FakeRuntime

        selection = resolve_selection('s100')
        task = ByteTrackTask.from_model(
            selection, tracker_factory=FakeTracker,
            runtime_factory=lambda path: FakeRuntime('s100'))
        tracks = task.predict(np.zeros((48, 64, 3), np.uint8))
        self.assertIsInstance(tracks, tuple)

    def test_split_forwarding_modules_are_removed(self):
        base = Path(__file__).resolve().parents[1] / 'runtime/python'
        for name in ('model_binding.py', 'model_runner.py'):
            self.assertFalse((base / name).exists(), name)
