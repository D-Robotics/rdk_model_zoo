"""Stage-contract tests for ClassificationTask (inference-contract §5)."""

from __future__ import annotations

import unittest

import numpy as np


def _make_task(target: str, logits: np.ndarray):
    from samples.vision.resnet.runtime.python.classification import ClassificationTask
    from samples.vision.resnet.runtime.python.model_binding import bind_model, resolve_selection
    from testsupport import runtime_metadata

    binding = bind_model(resolve_selection(target), runtime_metadata(target))
    calls: list[dict] = []

    def runner(inputs):
        calls.append(dict(inputs))
        return {binding.output_name: logits}

    return ClassificationTask(runner, binding, top_k=3), binding, calls


class StageContractTests(unittest.TestCase):
    def test_predict_equals_explicit_three_steps(self):
        logits = np.linspace(-5.0, 5.0, 1000, dtype=np.float32).reshape(1, 1000)
        task, _, _ = _make_task("x5", logits.reshape(1, 1000, 1, 1))
        image = np.full((60, 80, 3), 127, dtype=np.uint8)

        via_predict = task.predict(image)
        prepared = task.pre_process(image)
        outputs = task.forward(prepared.tensors)
        via_explicit = task.post_process(outputs)

        self.assertEqual(
            via_predict.class_ids.tolist(), via_explicit.class_ids.tolist())
        np.testing.assert_array_equal(via_predict.scores, via_explicit.scores)
        self.assertEqual(via_predict.labels, via_explicit.labels)

    def test_forward_returns_raw_runner_output_without_transformation(self):
        logits = np.zeros((1, 1000), dtype=np.float32)
        logits[0, 11] = 3.0
        logits[0, 12] = 1.5
        raw = logits.reshape(1, 1000, 1, 1)
        task, binding, _ = _make_task("x5", raw)
        prepared = task.pre_process(np.full((30, 40, 3), 90, dtype=np.uint8))

        outputs = task.forward(prepared.tensors)

        # forward adapts the container only: the validated raw tensor must be
        # bit-identical to the fixture — no softmax, no dequant, no scaling.
        np.testing.assert_array_equal(outputs[binding.output_name], raw)

    def test_interleaved_sizes_keep_per_call_context(self):
        logits = np.zeros((1, 1000), dtype=np.float32)
        task, _, _ = _make_task("s100", logits)
        wide = np.full((60, 80, 3), 30, dtype=np.uint8)
        small = np.full((20, 20, 3), 200, dtype=np.uint8)

        first = task.pre_process(wide)
        second = task.pre_process(small)
        third = task.pre_process(wide)

        self.assertEqual((first.transform.original_height, first.transform.original_width), (60, 80))
        self.assertEqual((second.transform.original_height, second.transform.original_width), (20, 20))
        self.assertEqual((third.transform.original_height, third.transform.original_width), (60, 80))
        # Letterbox geometry of the same input is reproducible and is not
        # overwritten by the interleaved different-size call.
        self.assertEqual(first.transform, third.transform)
        self.assertNotEqual(first.transform, second.transform)

    def test_call_delegates_to_predict(self):
        logits = np.zeros((1, 1000), dtype=np.float32)
        task, _, _ = _make_task("x5", logits.reshape(1, 1000, 1, 1))
        image = np.full((28, 34, 3), 60, dtype=np.uint8)

        via_call = task(image)
        via_predict = task.predict(image)

        self.assertEqual(
            via_call.class_ids.tolist(), via_predict.class_ids.tolist())
        np.testing.assert_array_equal(via_call.scores, via_predict.scores)


if __name__ == "__main__":
    unittest.main()
