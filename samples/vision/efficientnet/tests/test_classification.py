"""Policy-path tests for the EfficientNet classification task."""

from __future__ import annotations

import unittest

import numpy as np


class PolicyTests(unittest.TestCase):
    def test_softmax_policy_applies_stable_softmax_before_topk(self):
        """EfficientNet policy 'softmax': both source wrappers treat the
        graph output as logits and softmax it (X5 scipy.softmax; S the
        stable softmax inside get_topk_predictions), so the task applies the
        numerically stable softmax before Top-K."""

        from samples.vision.efficientnet.runtime.python.classification import (
            ClassificationTask,
        )
        from samples.vision.efficientnet.runtime.python.model_binding import (
            bind_model,
            resolve_selection,
        )
        from testsupport import runtime_metadata

        selection = resolve_selection("x5")
        binding = bind_model(selection, runtime_metadata("x5"))
        self.assertEqual(binding.contract.output_score_policy, "softmax")

        raw = np.zeros((1, 1000), dtype=np.float32)
        raw[0, 3] = 5.0
        raw[0, 11] = 3.0

        task = ClassificationTask(lambda _: {binding.output_name: raw}, binding, top_k=2)
        result = task.predict(np.zeros((20, 10, 3), dtype=np.uint8))

        self.assertEqual(result.class_ids.tolist(), [3, 11])
        shifted = np.exp(raw - raw.max(axis=-1, keepdims=True))
        expected = shifted / shifted.sum(axis=-1, keepdims=True)
        np.testing.assert_allclose(
            result.scores, expected[0, [3, 11]], rtol=1e-5
        )

    def test_injected_runner_flows_split_input_through_task_at_variant_geometry(self):
        from samples.vision.efficientnet.runtime.python.classification import ClassificationTask
        from samples.vision.efficientnet.runtime.python.model_binding import (
            bind_model,
            resolve_selection,
        )
        from testsupport import runtime_metadata

        # lite4 declares 380x380 on S boards; the prepared Y/UV planes must
        # follow that geometry, not a hardcoded 224.
        selection = resolve_selection("s100", variant="lite4")
        binding = bind_model(selection, runtime_metadata("s", 380))
        observed = {}
        scores = np.full((1, 1000), 0.0005, dtype=np.float32)
        scores[0, 42] = 0.99

        def runner(inputs):
            observed.update(inputs)
            return {binding.output_name: scores}

        result = ClassificationTask(runner, binding, top_k=1).predict(
            np.zeros((10, 20, 3), dtype=np.uint8)
        )
        self.assertEqual(result.class_ids.tolist(), [42])
        self.assertIn(binding.y_input_name, observed)
        self.assertIn(binding.uv_input_name, observed)
        self.assertEqual(
            tuple(observed[binding.y_input_name].shape), (1, 380, 380, 1)
        )
        self.assertEqual(
            tuple(observed[binding.uv_input_name].shape), (1, 190, 190, 2)
        )

    def test_packed_input_uses_canonical_flat_buffer(self):
        from samples.vision.efficientnet.runtime.python.classification import ClassificationTask
        from samples.vision.efficientnet.runtime.python.model_binding import (
            bind_model,
            resolve_selection,
        )
        from testsupport import runtime_metadata

        selection = resolve_selection("x5")
        binding = bind_model(selection, runtime_metadata("x5"))
        observed = {}
        scores = np.full((1, 1000), 0.0005, dtype=np.float32)
        scores[0, 5] = 0.99

        def runner(inputs):
            observed.update(inputs)
            return {binding.output_name: scores.reshape(1, 1000, 1, 1)}

        result = ClassificationTask(runner, binding, top_k=1).predict(
            np.zeros((10, 20, 3), dtype=np.uint8)
        )
        self.assertEqual(result.class_ids.tolist(), [5])
        # H2: packed NV12 feeds the canonical flat 1-D byte buffer.
        expected_bytes = binding.contract.input_height * 3 // 2 * binding.contract.input_width
        self.assertEqual(tuple(observed[binding.input_names[0]].shape), (expected_bytes,))


if __name__ == "__main__":
    unittest.main()
