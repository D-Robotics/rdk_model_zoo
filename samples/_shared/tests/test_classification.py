"""Policy-path and stage-isolation tests for the shared classification task.

The tests drive :class:`ClassificationTask` with an injected runner and a
synthetic binding, so they run on a host without a board SDK and prove the
score-policy semantics that the per-sample suites also assert through
their own binding tables.
"""

from __future__ import annotations

import unittest

import numpy as np

from samples._shared.classification import (
    ClassificationResult,
    ClassificationTask,
    topk_from_logits,
    topk_from_scores,
)
from samples._shared.cls_binding import (
    ClassificationContract,
    ModelBinding,
    ModelSelection,
)


def synthetic_binding(policy: str) -> ModelBinding:
    """A minimal S-style split-NV12 binding whose contract declares ``policy``."""

    contract = ClassificationContract(
        asset_id="s:fixture:fixture_32x32_nv12.hbm",
        variant="fixture",
        target="s100",
        model_format="hbm",
        input_protocol="split_nv12",
        input_height=32,
        input_width=32,
        class_count=1000,
        output_transform="raw_f32",
        output_semantics="synthetic_test_fixture",
        output_score_policy=policy,
        resize_type=0,
        resize_interpolation="linear",
        letterbox_interpolation="linear",
        source_manifest="docs/release/s/models.yaml",
    )
    selection = ModelSelection(
        asset_id=contract.asset_id,
        variant="fixture",
        target="s100",
        model_path="/nonexistent/fixture_32x32_nv12.hbm",
        contract=contract,
        sample_id="fixture",
    )
    return ModelBinding(
        selection=selection,
        contract=contract,
        model_name="fixture",
        input_names=("input_y", "input_uv"),
        input_shapes={
            "input_y": (1, 32, 32, 1),
            "input_uv": (1, 16, 16, 2),
        },
        output_name="output",
        output_shape=(1, 1000),
        output_dtype="float32",
        output_transform="raw_f32",
        output_quants={},
        y_input_name="input_y",
        uv_input_name="input_uv",
    )


def score_vector() -> np.ndarray:
    raw = np.full((1, 1000), 0.001, dtype=np.float32)
    raw[0, 3] = 0.9
    raw[0, 11] = 0.09
    return raw


class ScorePolicyTests(unittest.TestCase):
    def _task(self, policy):
        binding = synthetic_binding(policy)
        outputs = {"output": score_vector()}
        return ClassificationTask(lambda _tensors: outputs, binding, top_k=2), binding

    def test_policy_none_preserves_raw_values_exactly(self):
        task, _binding = self._task("none")
        result = task.predict(np.zeros((20, 10, 3), dtype=np.uint8))
        self.assertEqual(result.class_ids.tolist(), [3, 11])
        # float32 round-trip: no renormalisation of already-activated outputs.
        np.testing.assert_array_equal(
            result.scores, np.array([0.9, 0.09], dtype=np.float32)
        )

    def test_policy_softmax_applies_stable_softmax_before_topk(self):
        task, _binding = self._task("softmax")
        result = task.predict(np.zeros((20, 10, 3), dtype=np.uint8))
        raw = score_vector()
        shifted = np.exp(raw - raw.max(axis=-1, keepdims=True))
        expected = (shifted / shifted.sum(axis=-1, keepdims=True))[0, [3, 11]]
        np.testing.assert_allclose(result.scores, expected, rtol=1e-5)

    def test_policy_legacy_softmax_matches_softmax_numerically(self):
        # The pilot name keeps the warning semantics but the same math.
        task, _binding = self._task("legacy_softmax")
        result = task.predict(np.zeros((20, 10, 3), dtype=np.uint8))
        raw = score_vector()
        shifted = np.exp(raw - raw.max(axis=-1, keepdims=True))
        expected = (shifted / shifted.sum(axis=-1, keepdims=True))[0, [3, 11]]
        np.testing.assert_allclose(result.scores, expected, rtol=1e-5)

    def test_unknown_policy_is_a_visible_error(self):
        task, _binding = self._task("sigmoid")
        with self.assertRaises(Exception):
            task.predict(np.zeros((20, 10, 3), dtype=np.uint8))


class TopKHelperTests(unittest.TestCase):
    def test_topk_from_scores_softmax_false_preserves_raw(self):
        raw = np.array([0.5, 0.2, 0.3], dtype=np.float32)
        result = topk_from_scores(raw, 3, softmax=False)
        self.assertEqual(result.class_ids.tolist(), [0, 2, 1])
        np.testing.assert_array_equal(
            result.scores, np.array([0.5, 0.3, 0.2], dtype=np.float32)
        )

    def test_topk_from_logits_softmaxes(self):
        logits = np.array([[0.0, 2.0, 1.0]], dtype=np.float32)
        result = topk_from_logits(logits, 2)
        self.assertEqual(result.class_ids.tolist(), [1, 2])
        self.assertGreater(float(result.scores[0]), float(result.scores[1]))
        self.assertLess(float(result.scores[0]), 1.0)


class StageIsolationTests(unittest.TestCase):
    """Two interleaved sizes must not leak geometry between calls."""

    def test_interleaved_sizes_keep_their_own_transforms(self):
        binding = synthetic_binding("none")
        outputs = {"output": score_vector()}
        seen = []
        task = ClassificationTask(
            lambda tensors: (seen.append(tensors), outputs)[1], binding, top_k=1
        )
        small = task.pre_process(np.zeros((12, 34, 3), dtype=np.uint8))
        large = task.pre_process(np.zeros((60, 21, 3), dtype=np.uint8))
        # Each PreparedInput carries its own per-call transform; the binding
        # state is untouched by either call.
        self.assertNotEqual(small.transform, large.transform)
        self.assertEqual(binding.input_names, ("input_y", "input_uv"))
        # Forward consumes only the prepared tensors of that call.
        first = task.forward(small)
        second = task.forward(large)
        self.assertEqual(len(seen), 2)
        self.assertIs(first, outputs)
        self.assertIs(second, outputs)

    def test_predict_equals_explicit_three_stages(self):
        binding = synthetic_binding("none")
        outputs = {"output": score_vector()}
        task = ClassificationTask(lambda _t: outputs, binding, top_k=2)
        image = np.zeros((17, 9, 3), dtype=np.uint8)
        chained = task.predict(image)
        prepared = task.pre_process(image)
        explicit = task.post_process(task.forward(prepared))
        np.testing.assert_array_equal(chained.class_ids, explicit.class_ids)
        np.testing.assert_array_equal(chained.scores, explicit.scores)
        self.assertEqual(chained.labels, explicit.labels)


class ClassificationResultTests(unittest.TestCase):
    def test_legacy_surface_is_consistent(self):
        scores = np.array([0.7, 0.3], dtype=np.float32)
        result = ClassificationResult(
            class_ids=np.array([3, 11]), scores=scores, labels=("a", "b")
        )
        np.testing.assert_array_equal(result.topk_idx, np.array([3, 11]))
        np.testing.assert_array_equal(result.topk_prob, scores)
        self.assertEqual(result.topk_labels, ("a", "b"))
        idx, prob, labels = result.as_legacy_tuple()
        np.testing.assert_array_equal(idx, np.array([3, 11]))
        self.assertEqual(labels, ("a", "b"))
        unpacked = list(result)
        self.assertEqual(len(unpacked), 3)


if __name__ == "__main__":
    unittest.main()
