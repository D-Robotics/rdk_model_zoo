"""Numerical and input-contract tests for the reusable classification task."""

from __future__ import annotations

import unittest

import numpy as np


class ClassificationTests(unittest.TestCase):
    def test_fixed_logits_return_descending_top_k_probabilities(self):
        from samples.vision.resnet.runtime.python.classification import (
            topk_from_logits,
        )

        result = topk_from_logits(np.array([0.0, 2.0, -1.0, 1.0], dtype=np.float32), 3)

        self.assertEqual(result.class_ids.tolist(), [1, 3, 0])
        self.assertTrue(np.all(np.diff(result.scores) <= 0))
        self.assertAlmostEqual(float(result.scores.sum()), 0.9679414, places=5)

    def test_invalid_logits_and_top_k_fail_loudly(self):
        from samples.vision.resnet.runtime.python.classification import topk_from_logits

        with self.assertRaises(ValueError):
            topk_from_logits(np.array([[1.0, np.nan]], dtype=np.float32), 1)
        with self.assertRaises(ValueError):
            topk_from_logits(np.array([1.0, 2.0]), 0)
        with self.assertRaises(ValueError):
            topk_from_logits(np.ones((2, 2), dtype=np.float32), 1)

    def test_injected_runner_keeps_one_task_flow_for_packed_input(self):
        from samples.vision.resnet.runtime.python.classification import ClassificationTask
        from samples.vision.resnet.runtime.python.model_binding import bind_model, resolve_selection
        from testsupport import runtime_metadata

        selection = resolve_selection("x5")
        binding = bind_model(selection, runtime_metadata("x5"))
        observed = {}

        logits = np.full((1, 1000), -10.0, dtype=np.float32)
        logits[0, 7] = 4.0

        def runner(inputs):
            observed.update(inputs)
            return {binding.output_name: logits.reshape(1, 1000, 1, 1)}

        task = ClassificationTask(runner, binding, top_k=1)
        result = task.predict(np.zeros((10, 20, 3), dtype=np.uint8))

        self.assertEqual(result.class_ids.tolist(), [7])
        # H2: packed NV12 feeds the canonical flat 1-D byte buffer.
        self.assertEqual(tuple(observed[binding.input_names[0]].shape), (224 * 336,))

    def test_injected_runner_rejects_wrong_output_shape(self):
        from samples.vision.resnet.runtime.python.classification import ClassificationTask
        from samples.vision.resnet.runtime.python.model_binding import bind_model, resolve_selection
        from testsupport import runtime_metadata

        binding = bind_model(resolve_selection("x5"), runtime_metadata("x5"))
        wrong = np.zeros((1, 5), dtype=np.float32)
        with self.assertRaises(ValueError):
            ClassificationTask(
                lambda _: {binding.output_name: wrong}, binding, top_k=1
            ).predict(np.zeros((20, 20, 3), dtype=np.uint8))

    def test_injected_runner_rejects_wrong_class_count_and_dtype(self):
        from samples.vision.resnet.runtime.python.classification import ClassificationTask
        from samples.vision.resnet.runtime.python.model_binding import bind_model, resolve_selection
        from testsupport import runtime_metadata

        binding = bind_model(resolve_selection("x5"), runtime_metadata("x5"))
        wrong = np.zeros((1, 1001, 1, 1), dtype=np.float32)
        with self.assertRaises(ValueError):
            ClassificationTask(
                lambda _: {binding.output_name: wrong}, binding, top_k=1
            ).predict(np.zeros((20, 20, 3), dtype=np.uint8))

        wrong_dtype = np.zeros((1, 1000, 1, 1), dtype=np.int8)
        with self.assertRaises(ValueError):
            ClassificationTask(
                lambda _: {binding.output_name: wrong_dtype}, binding, top_k=1
            ).predict(np.zeros((20, 20, 3), dtype=np.uint8))

    def test_declared_dequant_transform_is_executed_by_post_process(self):
        # H1 full chain: a contract declaring 'dequant' makes post_process
        # dequantize the runner's int8 output with the binding's descriptor
        # before the legacy softmax policy.  Synthetic fixture, not board
        # evidence; the published ResNet artifacts declare raw_f32.
        import dataclasses

        from samples.vision.resnet.runtime.python.classification import ClassificationTask
        from samples.vision.resnet.runtime.python.model_binding import bind_model, resolve_selection
        from testsupport import runtime_metadata

        class _QuantInfo:
            def __init__(self):
                import numpy as np

                class _EnumLike:
                    name = "SCALE"

                self.quant_type = _EnumLike()
                self.scale = np.asarray(0.5, dtype=np.float32)
                self.zero_point = np.asarray(2.0, dtype=np.float32)
                self.axis = 0

        selection = resolve_selection("x5")
        dequant_selection = dataclasses.replace(
            selection,
            contract=dataclasses.replace(
                selection.contract, output_transform="dequant"
            ),
        )
        descriptor = _QuantInfo()
        facts = runtime_metadata("x5")
        facts = type(facts).from_mapping(
            {
                "model_name": facts.model_name,
                "input_names": facts.input_names,
                "input_shapes": facts.input_shapes,
                "input_dtypes": facts.input_dtypes,
                "output_names": facts.output_names,
                "output_shapes": facts.output_shapes,
                "output_dtypes": {facts.output_names[0]: "int8"},
                "output_quants": {facts.output_names[0]: descriptor},
            }
        )
        binding = bind_model(dequant_selection, facts)

        import numpy as np

        # q=10 -> (10-2)*0.5 = 4.0 at class 7; all others map below it.
        raw = np.zeros((1, 1000, 1, 1), dtype=np.int8)
        raw[0, 7, 0, 0] = 10
        raw[0, 0, 0, 0] = 4

        task = ClassificationTask(lambda _: {binding.output_name: raw}, binding, top_k=1)
        result = task.predict(np.zeros((10, 20, 3), dtype=np.uint8))
        self.assertEqual(result.class_ids.tolist(), [7])

    def test_split_input_keeps_y_and_uv_as_separate_tensors(self):
        from samples.vision.resnet.runtime.python.classification import ClassificationTask
        from samples.vision.resnet.runtime.python.model_binding import bind_model, resolve_selection
        from testsupport import runtime_metadata

        selection = resolve_selection("s100")
        binding = bind_model(selection, runtime_metadata("s"))
        observed = {}

        logits = np.zeros((1, 1000), dtype=np.float32)

        def runner(inputs):
            observed.update(inputs)
            return {binding.output_name: logits}

        ClassificationTask(runner, binding, top_k=1).predict(
            np.zeros((20, 10, 3), dtype=np.uint8))

        self.assertEqual(tuple(observed[binding.y_input_name].shape), (1, 224, 224, 1))
        self.assertEqual(tuple(observed[binding.uv_input_name].shape), (1, 112, 112, 2))

    def test_non_image_input_is_rejected_before_runner(self):
        from samples.vision.resnet.runtime.python.classification import ClassificationTask
        from samples.vision.resnet.runtime.python.model_binding import bind_model, resolve_selection
        from testsupport import runtime_metadata

        selection = resolve_selection("x5")
        binding = bind_model(selection, runtime_metadata("x5"))

        with self.assertRaises(ValueError):
            ClassificationTask(lambda _: {}, binding).predict(np.zeros((224, 224), dtype=np.uint8))


if __name__ == "__main__":
    unittest.main()
