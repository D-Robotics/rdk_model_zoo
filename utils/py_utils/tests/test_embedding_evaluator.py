# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Independent semantic checks reject constant or unrelated encoder outputs."""
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from utils.py_utils.embedding_evaluator import evaluate_relation, evaluate_token_correspondence, image_vector, main


class RelationTests(unittest.TestCase):
    def features(self, positive=(0.9, 0.1), negative=(0, 1)):
        return {"anchor": np.array([[1., 0.]]), "repeat": np.array([[1., 0.]]),
                "mild": np.array([[1., 0.05]]), "positive": np.array([positive], dtype=np.float64),
                "negative": np.array([negative], dtype=np.float64)}

    def test_independent_relation_and_stability(self):
        self.assertTrue(evaluate_relation(self.features())["passed"])

    def test_constant_features_fail_semantics(self):
        result = evaluate_relation(self.features((1, 0), (1, 0)))
        self.assertFalse(result["passed"])
        self.assertTrue(result["checks"]["repeat"])

    def test_reversed_ranking_fails(self):
        self.assertFalse(evaluate_relation(self.features((0, 1), (1, 0)))["passed"])

    def test_patch_mean_not_flattened_spatial_matching(self):
        np.testing.assert_allclose(image_vector(np.array([[[1., 0.], [0., 1.]]])),
                                   np.array([1., 1.]) / np.sqrt(2))

    def test_invalid_features_fail(self):
        for value in (np.zeros((1, 2)), np.array([[np.nan, 1.]]),
                      np.ones((1, 2), dtype=np.int32), np.ones((2, 2))):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    image_vector(value)

    def token_features(self):
        anchor = np.eye(4, dtype=np.float64)[None]
        # Different image semantics may fail mean pooling while the original
        # image's spatial token response remains stable and discriminative.
        return {"anchor": anchor, "repeat": anchor.copy(), "mild": anchor.copy(),
                "positive": anchor.copy(), "negative": -anchor.copy()}

    def test_aligned_tokens_keep_mean_diagnostic_non_gating(self):
        features = self.token_features()
        features["positive"] = -features["anchor"]
        result = evaluate_token_correspondence(features)
        self.assertTrue(result["passed"])
        self.assertFalse(result["mean_image_relation_diagnostic"]["passed"])
        self.assertFalse(result["mean_image_relation_diagnostic"]["affects_pass_criterion"])
        self.assertFalse(result["semantic_image_retrieval"])

    def test_shuffled_patch_positions_fail_aligned_mild_check(self):
        features = self.token_features()
        features["mild"] = features["mild"][:, [1, 2, 3, 0], :]
        self.assertFalse(evaluate_token_correspondence(features)["passed"])

    def test_constant_patch_output_fails_discrimination(self):
        features = {key: np.ones((1, 4, 4)) for key in self.token_features()}
        self.assertFalse(evaluate_token_correspondence(features)["passed"])

    def test_corrupted_patch_response_fails(self):
        features = self.token_features()
        features["repeat"] = -features["repeat"]
        self.assertFalse(evaluate_token_correspondence(features)["passed"])

    def test_role_numeric_failure_still_saves_raw_and_report(self):
        import cv2
        from utils.py_utils.runtime_meta import RuntimeMetadata
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            paths = []
            for i in range(3):
                path = directory / f"{i}.png"
                cv2.imwrite(str(path), np.full((2, 2, 3), i, dtype=np.uint8))
                paths.append(path)
            modelpath = directory / "model.hbm"
            modelpath.write_bytes(b"fixture")
            model = SimpleNamespace(runner=SimpleNamespace(metadata=RuntimeMetadata.from_mapping({
                "model_name": "test", "input_names": [], "output_names": []})),
                predict=lambda image: np.ones((1, 2), dtype=np.int32))
            args = ["--target", "s100", "--asset-id", "test", "--model-path", str(modelpath),
                    "--anchor", str(paths[0]), "--positive", str(paths[1]), "--negative", str(paths[2]),
                    "--output-dir", str(directory / "out")]
            with patch("builtins.print"):
                self.assertEqual(main(lambda args: [("role", model)], args), 1)
            report = json.loads((directory / "out/result.json").read_text())
            self.assertIn("dequantize", report["roles"]["role"]["error"])
            self.assertTrue((directory / "out/role.npz").is_file())


if __name__ == "__main__":
    unittest.main()
