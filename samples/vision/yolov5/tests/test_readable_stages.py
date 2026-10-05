"""The readable stage surface (preprocess/infer/postprocess) on YOLOv5Task.

These tests pin the accepted readable-runtime contract: the new stage names
carry the real implementation, the established ``pre_process``/``forward``/
``post_process`` names delegate to them (no second implementation),
``predict`` equals the explicit three-step chain with this call's context,
one ``predict`` is exactly one runner call, and consecutive different-size
images keep their own geometry on both target protocols. All execution uses
the injected host fixture from ``test_yolov5``; no board SDK is loaded and no
board inference is claimed.
"""

from unittest import mock
import unittest
import numpy as np

from samples._shared.runtime_meta import RuntimeMetadata
from samples.vision.yolov5.runtime.python.model_binding import bind_model, resolve_selection
from samples.vision.yolov5.runtime.python.detection import YOLOv5Task
from test_yolov5 import FakeRuntime


def _task(target='x5'):
    runtime = FakeRuntime(target)
    binding = bind_model(
        resolve_selection(target), RuntimeMetadata.from_mapping(runtime.facts))
    calls = []

    def runner(tensors):
        calls.append(tensors)
        return runtime.outputs

    return YOLOv5Task(runner, binding), calls, binding


class Yolov5ReadableStageTests(unittest.TestCase):
    def test_new_stage_names_and_predict_equality(self):
        for target in ('x5', 's100'):
            with self.subTest(target=target):
                task, calls, _ = _task(target)
                image = np.zeros((97, 151, 3), np.uint8)
                prepared = task.preprocess(image)
                outputs = task.infer(prepared.tensors)
                staged = task.postprocess(outputs, prepared.context)
                predicted = task.predict(image)
                for name in ('boxes', 'scores', 'class_ids'):
                    np.testing.assert_array_equal(
                        getattr(staged, name), getattr(predicted, name))
                self.assertGreater(len(predicted.scores), 0)
                self.assertEqual(len(calls), 2)

    def test_legacy_stage_names_delegate_to_the_new_methods(self):
        task, _, _ = _task('x5')
        image = np.zeros((97, 151, 3), np.uint8)
        with mock.patch.object(task, 'preprocess', wraps=task.preprocess) as pre, \
                mock.patch.object(task, 'infer', wraps=task.infer) as infer, \
                mock.patch.object(task, 'postprocess', wraps=task.postprocess) as post:
            prepared = task.pre_process(image)
            task.post_process(task.forward(prepared.tensors), prepared.context)
        pre.assert_called_once()
        infer.assert_called_once()
        post.assert_called_once()

    def test_predict_executes_the_runner_exactly_once(self):
        task, calls, _ = _task('x5')
        task.predict(np.zeros((37, 59, 3), np.uint8))
        self.assertEqual(len(calls), 1)

    def test_different_size_images_keep_their_own_geometry(self):
        task, _, _ = _task('s100')
        wide = np.full((200, 300, 3), 30, np.uint8)
        small = np.full((40, 28, 3), 200, np.uint8)
        for image in (wide, small):
            prepared = task.preprocess(image)
            staged = task.postprocess(task.infer(prepared.tensors), prepared.context)
            predicted = task.predict(image)
            for name in ('boxes', 'scores', 'class_ids'):
                np.testing.assert_array_equal(
                    getattr(staged, name), getattr(predicted, name))


if __name__ == '__main__':
    unittest.main()
