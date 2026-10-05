"""The readable stage surface (preprocess/infer/postprocess) on the YOLOE task.

These tests pin the accepted readable-runtime contract for ``YOLOE``: the new
stage names carry the real implementation, the established
``pre_process``/``forward``/``post_process`` names delegate to them (no second
implementation), ``predict`` equals the explicit three-step chain carrying this
image's context, one ``predict`` is exactly one runner call, and consecutive
different-size images keep their own geometry. All execution uses the injected
host fixture from ``test_runtime.fake_runtime``; no board SDK is loaded and no
board inference is claimed.
"""

from types import SimpleNamespace
from unittest import mock
import unittest
import numpy as np

from test_runtime import fake_runtime
from samples.vision.yoloe.runtime.python.model_binding import resolve_selection
from samples.vision.yoloe.runtime.python.model_runner import build_runner
from samples.vision.yoloe.runtime.python.yoloe import YOLOE


def _task(target, variant):
    selection = resolve_selection(target, variant=variant)
    runtime, _ = fake_runtime(selection)
    runner = build_runner(
        selection,
        runtime_loader=lambda: SimpleNamespace(HB_HBMRuntime=lambda p: runtime),
    )
    return YOLOE(selection, runner=runner), runtime


_TASKS = [
    ("x5", "11s"),
    ("s100", "11s"),
    ("s100p", "26n"),
]


class YoloEReadableStageTests(unittest.TestCase):
    def test_new_stage_names_and_predict_equality(self):
        for target, variant in _TASKS:
            with self.subTest(target=target, variant=variant):
                task, _ = _task(target, variant)
                image = np.zeros((320, 640, 3), np.uint8)
                prepared = task.preprocess(image)
                raw = task.infer(prepared)
                staged = task.postprocess(raw, prepared.context)
                predicted = task.predict(image)
                for name in ("boxes", "scores", "class_ids"):
                    np.testing.assert_array_equal(
                        getattr(staged, name), getattr(predicted, name))
                self.assertEqual(staged.mask_layout, predicted.mask_layout)
                self.assertGreater(len(predicted.boxes), 0)

    def test_legacy_stage_names_delegate_to_the_new_methods(self):
        task, _ = _task("x5", "11s")
        image = np.zeros((320, 640, 3), np.uint8)
        with mock.patch.object(task, "preprocess", wraps=task.preprocess) as pre, \
                mock.patch.object(task, "infer", wraps=task.infer) as infer, \
                mock.patch.object(task, "postprocess", wraps=task.postprocess) as post:
            prepared = task.pre_process(image)
            task.post_process(task.forward(prepared), prepared.context)
        pre.assert_called_once()
        infer.assert_called_once()
        post.assert_called_once()

    def test_predict_executes_the_runner_exactly_once(self):
        for target, variant in _TASKS:
            with self.subTest(target=target, variant=variant):
                task, runtime = _task(target, variant)
                before = runtime.calls
                task.predict(np.zeros((37, 59, 3), np.uint8))
                self.assertEqual(runtime.calls, before + 1)

    def test_different_size_images_keep_their_own_geometry(self):
        task, _ = _task("s100", "11s")
        wide = np.full((200, 300, 3), 30, np.uint8)
        small = np.full((40, 28, 3), 200, np.uint8)
        for image in (wide, small):
            prepared = task.preprocess(image)
            staged = task.postprocess(task.infer(prepared), prepared.context)
            predicted = task.predict(image)
            for name in ("boxes", "scores", "class_ids"):
                np.testing.assert_array_equal(
                    getattr(staged, name), getattr(predicted, name))


if __name__ == "__main__":
    unittest.main()
