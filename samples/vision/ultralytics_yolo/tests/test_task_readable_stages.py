"""The readable stage surface on every non-detect YOLO task class.

``detect.py`` is the accepted exemplar; these tests pin the same contract for
the cls/seg/pose/OBB and YOLO26 task classes: the new stage names carry the
real implementation, the established ``pre_process``/``forward``/``post_process``
names delegate to them (no second implementation), ``predict`` equals the
explicit three-step chain with per-call geometry, and one ``predict`` is
exactly one model call. All runners are injected host fixtures; no board SDK
is loaded and no board inference is claimed.
"""

from unittest import mock
import unittest
import numpy as np

from test_forward_purity import fixture as detection_fixture
from test_classification_binding import fixture as classification_fixture
from test_segmentation_binding import fixture as segmentation_fixture
from test_pose_binding import fixture as pose_fixture
from test_obb_binding import fixture as obb_fixture

from samples.vision.ultralytics_yolo.runtime.python.yolo26_det import (
    YOLO26Detect,
    YOLO26DetectConfig,
)


def _cls_task():
    return classification_fixture()[0]


def _seg_task():
    return segmentation_fixture()[0]


def _pose_task():
    return pose_fixture()[0]


def _ltrb_task():
    runner, _, _, contract = detection_fixture(sdk=True, ltrb=True)
    return YOLO26Detect(
        YOLO26DetectConfig(
            "fixture.bin", classes_num=1, contract=contract, nms_thres=0.45
        ),
        runner=runner,
    )


def _obb_task():
    return obb_fixture()[0]


_TASKS = {
    "cls": _cls_task,
    "seg": _seg_task,
    "pose": _pose_task,
    "yolo26_det": _ltrb_task,
    "yolo26_obb": _obb_task,
}


def _assert_results_equal(test, actual, expected):
    """Compare task results element-wise, nested arrays and masks included."""
    if isinstance(actual, np.ndarray) or isinstance(expected, np.ndarray):
        np.testing.assert_array_equal(actual, expected)
    elif isinstance(actual, (list, tuple)) and isinstance(expected, (list, tuple)):
        test.assertEqual(len(actual), len(expected))
        for left, right in zip(actual, expected):
            _assert_results_equal(test, left, right)
    elif isinstance(actual, dict) and isinstance(expected, dict):
        test.assertEqual(actual.keys(), expected.keys())
        for key in actual:
            _assert_results_equal(test, actual[key], expected[key])
    else:
        test.assertEqual(actual, expected)


def _explicit_chain(task, image):
    """preprocess → infer → postprocess with this call's own geometry."""
    prepared = task.preprocess(image)
    outputs = task.infer(prepared)
    transform = getattr(prepared, "transform", None)
    if transform is None:
        return task.postprocess(outputs)
    return task.postprocess(outputs, transform=transform)


class ReadableStageSurface(unittest.TestCase):
    def test_every_task_exposes_the_new_stage_names(self):
        image = np.zeros((37, 59, 3), np.uint8)
        for name, build in _TASKS.items():
            with self.subTest(task=name):
                task = build()
                _assert_results_equal(self, task.predict(image),
                                      _explicit_chain(task, image))

    def test_legacy_stage_names_delegate_to_the_new_methods(self):
        image = np.zeros((37, 59, 3), np.uint8)
        for name, build in _TASKS.items():
            with self.subTest(task=name):
                task = build()
                prepared = task.preprocess(image)
                with mock.patch.object(
                    task, "preprocess", wraps=task.preprocess
                ) as pre, mock.patch.object(
                    task, "infer", wraps=task.infer
                ) as infer, mock.patch.object(
                    task, "postprocess", wraps=task.postprocess
                ) as post:
                    legacy_prepared = task.pre_process(image)
                    legacy_outputs = task.forward(legacy_prepared)
                    transform = getattr(legacy_prepared, "transform", None)
                    if transform is None:
                        task.post_process(legacy_outputs)
                    else:
                        task.post_process(legacy_outputs, transform=transform)
                pre.assert_called()
                infer.assert_called()
                post.assert_called()

    def test_predict_executes_the_model_exactly_once(self):
        image = np.zeros((64, 64, 3), np.uint8)
        for name, build in _TASKS.items():
            with self.subTest(task=name):
                task = build()
                with mock.patch.object(
                    task.model, "run", wraps=task.model.run
                ) as spy:
                    task.predict(image)
                self.assertEqual(spy.call_count, 1)

    def test_different_size_images_keep_their_own_geometry(self):
        wide = np.full((200, 300, 3), 30, np.uint8)
        small = np.full((40, 28, 3), 200, np.uint8)
        for name, build in _TASKS.items():
            with self.subTest(task=name):
                task = build()
                for image in (wide, small):
                    _assert_results_equal(self, task.predict(image),
                                          _explicit_chain(task, image))

    def test_cls_topk_reaches_the_explicit_postprocess(self):
        task = _cls_task()
        image = np.zeros((51, 29, 3), np.uint8)
        self.assertEqual(len(task.predict(image)), 5)
        prepared = task.preprocess(image)
        self.assertEqual(len(task.postprocess(task.infer(prepared), topk=3)), 3)


if __name__ == "__main__":
    unittest.main()
