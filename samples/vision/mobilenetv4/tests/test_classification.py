"""Policy-path tests for the MobileNetV4 classification task."""

from __future__ import annotations

import unittest

import numpy as np


class PolicyTests(unittest.TestCase):
    def test_mobilenetv4_policy_preserves_or_transforms_scores_as_declared(self):
        """v3/v4 policy 'softmax': the task applies a stable softmax to the
        graph's raw logits before Top-K."""

        from utils.py_utils.classification import (
            ClassificationTask,
            topk_from_scores,
        )
        from samples.vision.mobilenetv4.runtime.python.cli import (
            bind_model,
            resolve_selection,
        )
        from testsupport import runtime_metadata

        target = "x5"
        protocol = "x5" if target == "x5" else "s"
        selection = resolve_selection(target, variant="small")
        binding = bind_model(selection, runtime_metadata(protocol, variant="small"))
        policy = binding.contract.output_score_policy

        raw = np.full((1, 1000), 0.001, dtype=np.float32)
        raw[0, 3] = 0.9
        raw[0, 11] = 0.09

        task = ClassificationTask(lambda _: {binding.output_name: raw}, binding, top_k=2)
        result = task.predict(np.zeros((20, 10, 3), dtype=np.uint8))

        self.assertEqual(result.class_ids.tolist(), [3, 11])
        if policy == "none":
            # v1/v2: the graph already emits probabilities; raw values are
            # preserved exactly (float32 round-trip), no renormalisation.
            np.testing.assert_array_equal(
                result.scores, np.array([0.9, 0.09], dtype=np.float32)
            )
        else:
            # v3/v4: logits get a stable softmax before Top-K; the top-k
            # slice need not sum to one, so compare against the stable
            # softmax computed here.
            shifted = np.exp(raw - raw.max(axis=-1, keepdims=True))
            expected = shifted / shifted.sum(axis=-1, keepdims=True)
            np.testing.assert_allclose(
                result.scores, expected[0, [3, 11]], rtol=1e-5
            )

    def test_published_preprocessing_feeds_the_evaluation_bytes(self):
        """The runtime applies the geometry the models were evaluated with.

        The published contract resizes the shorter edge and center-crops (type 2).
        Preparing the host evaluation crop and feeding it through a plain direct
        resize (a no-op at 224x224) must yield identical NV12 tensors, for both
        variants and both input protocols.
        """

        try:
            from PIL import Image
        except ImportError:
            self.skipTest("Pillow is required for the shorter-edge center crop")

        from samples.vision.mobilenetv4.runtime.python.cli import (
            bind_model,
            resolve_selection,
        )
        from utils.py_utils.classification_host import prepare_rgb
        from utils.py_utils.tensor_io import prepare_nv12
        from testsupport import runtime_metadata

        rng = np.random.default_rng(7)
        bgr = rng.integers(0, 256, size=(300, 400, 3), dtype=np.uint8)
        for variant, crop_pct in (("small", 0.875), ("medium", 0.95)):
            contract = {
                "size": 224, "crop_pct": crop_pct, "geometry": "resize_shorter_center_crop",
                "interpolation": "pil_bicubic", "color": "RGB", "layout": "NCHW",
                "dtype": "float32", "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225], "batch": 1, "class_count": 1000,
                "output": "logits",
            }
            evaluation_crop = prepare_rgb(Image.fromarray(bgr[:, :, ::-1]), contract)[:, :, ::-1]
            for target in ("x5", "s100", "s100p", "s600"):
                with self.subTest(variant=variant, target=target):
                    binding = bind_model(
                        resolve_selection(target, variant=variant),
                        runtime_metadata("x5" if target == "x5" else "s", variant=variant))
                    self.assertEqual(binding.contract.resize_type, 2)
                    published = prepare_nv12(bgr, binding)
                    evaluated = prepare_nv12(
                        np.ascontiguousarray(evaluation_crop), binding, resize_type=0)
                    self.assertEqual(set(published.tensors), set(evaluated.tensors))
                    for name, tensor in published.tensors.items():
                        np.testing.assert_array_equal(tensor, evaluated.tensors[name])

    def test_topk_from_scores_softmax_false_preserves_raw_values(self):
        from utils.py_utils.classification import topk_from_scores

        raw = np.array([0.5, 0.2, 0.3], dtype=np.float32)
        result = topk_from_scores(raw, 3, softmax=False)
        self.assertEqual(result.class_ids.tolist(), [0, 2, 1])
        np.testing.assert_array_equal(
            result.scores, np.array([0.5, 0.3, 0.2], dtype=np.float32)
        )

    def test_injected_runner_flows_split_input_through_task(self):
        from utils.py_utils.classification import ClassificationTask
        from samples.vision.mobilenetv4.runtime.python.cli import (
            bind_model,
            resolve_selection,
        )
        from testsupport import runtime_metadata

        selection = resolve_selection("s100", variant="small")
        binding = bind_model(selection, runtime_metadata("s", variant="small"))
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

    def test_packed_input_uses_canonical_flat_buffer(self):
        from utils.py_utils.classification import ClassificationTask
        from samples.vision.mobilenetv4.runtime.python.cli import (
            bind_model,
            resolve_selection,
        )
        from testsupport import runtime_metadata

        selection = resolve_selection("x5", variant="small")
        binding = bind_model(selection, runtime_metadata("x5", variant="small"))
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
