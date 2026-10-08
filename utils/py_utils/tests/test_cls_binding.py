"""Host tests for the shared classification binding machinery.

These tests cover the target-independent invariants (rank rule, vector
normalisation, score-policy vocabulary).  The manifest-driven paths are
exercised end to end by each sample's suite against the real manifests.
"""

from __future__ import annotations

import unittest
from pathlib import Path
import tempfile
from types import SimpleNamespace

import numpy as np

from utils.py_utils.cls_binding import (
    SCORE_POLICIES,
    MetadataMismatchError,
    RuntimeMetadata,
    VariantFacts,
    normalise_score_vector,
    score_vector_shape,
)


class ScoreVectorShapeTests(unittest.TestCase):
    """The H4 rank rule: squeeze singletons, then require the class count."""

    def test_all_source_spellings_of_1000_bind(self):
        for shape in [(1000,), (1, 1000), (1, 1000, 1), (1, 1000, 1, 1)]:
            self.assertTrue(
                score_vector_shape(shape, 1000), f"shape {shape} should bind"
            )

    def test_wrong_class_count_is_rejected(self):
        self.assertFalse(score_vector_shape((1, 1001), 1000))
        self.assertFalse(score_vector_shape((1000,), 100))

    def test_real_extra_dimension_is_rejected_not_flattened(self):
        self.assertFalse(score_vector_shape((1, 1000, 2), 1000))
        self.assertFalse(score_vector_shape((2, 1000), 1000))

    def test_empty_and_invalid_shapes_are_rejected(self):
        self.assertFalse(score_vector_shape(None, 1000))
        self.assertFalse(score_vector_shape((), 1000))
        self.assertFalse(score_vector_shape((0, 1000), 1000))


class NormaliseScoreVectorTests(unittest.TestCase):
    def test_x5_and_s_spellings_squeeze_to_the_same_vector(self):
        flat = np.arange(1000, dtype=np.float32)
        x5 = flat.reshape(1, 1000, 1, 1)
        s = flat.reshape(1, 1000)
        for candidate in (flat, x5, s):
            np.testing.assert_array_equal(normalise_score_vector(candidate), flat)

    def test_ambiguous_layout_raises(self):
        with self.assertRaises(MetadataMismatchError):
            normalise_score_vector(np.zeros((2, 1000), dtype=np.float32))


class PolicyVocabularyTests(unittest.TestCase):
    def test_score_policies_are_the_declared_three(self):
        self.assertEqual(SCORE_POLICIES, ("legacy_softmax", "softmax", "none"))

    def test_variant_facts_defaults_match_the_shared_classification_shape(self):
        facts = VariantFacts(input_height=224, input_width=224)
        self.assertEqual(facts.class_count, 1000)
        self.assertEqual(facts.output_transform, "raw_f32")
        self.assertEqual(facts.output_score_policy, "legacy_softmax")
        self.assertEqual(facts.resize_type, 1)
        self.assertEqual(facts.resize_interpolation, "linear")
        self.assertEqual(facts.letterbox_interpolation, "linear")

    def test_variant_facts_accepts_the_mobileNet_fact_spellings(self):
        # The B1 mobilenet tables spell the facts exactly like this; the
        # dataclass must keep accepting them.
        facts = VariantFacts(
            input_height=256,
            input_width=256,
            output_score_policy="softmax",
            resize_type=0,
            resize_interpolation="linear",
        )
        self.assertEqual((facts.input_height, facts.input_width), (256, 256))
        self.assertEqual(facts.output_score_policy, "softmax")


class RuntimeMetadataRankTests(unittest.TestCase):
    def test_from_mapping_keeps_multi_output_shapes(self):
        metadata = RuntimeMetadata.from_mapping(
            {
                "model_name": "fixture",
                "input_names": ["input_y", "input_uv"],
                "input_shapes": {
                    "input_y": (1, 224, 224, 1),
                    "input_uv": (1, 112, 112, 2),
                },
                "input_dtypes": {"input_y": "U8", "input_uv": "U8"},
                "output_names": ["output"],
                "output_shapes": {"output": (1, 1000)},
                "output_dtypes": {"output": "F32"},
            }
        )
        self.assertTrue(score_vector_shape(metadata.output_shapes["output"], 1000))


class LocalFileRunnerTests(unittest.TestCase):
    def test_arbitrary_classifier_binds_without_sample_catalog(self):
        from utils.py_utils.model_runner import RuntimeModelRunner

        runtime = SimpleNamespace(
            model_names=["tiny"], input_names={"tiny": ["pixels"]},
            input_shapes={"tiny": {"pixels": (1, 3, 8, 8)}},
            input_dtypes={"tiny": {"pixels": "U8"}},
            output_names={"tiny": ["scores"]}, output_shapes={"tiny": {"scores": (1, 3)}},
            output_dtypes={"tiny": {"scores": "F32"}},
            run=lambda payload: {"tiny": {"scores": np.array([[1, 2, 3]], dtype=np.float32)}},
        )
        runner = RuntimeModelRunner.from_file(
            "tiny.bin", target="x5", input_size=(8, 8), class_count=3, runtime=runtime)
        self.assertFalse(runner.loaded)
        binding = runner.load()
        self.assertEqual(binding.contract.class_count, 3)
        self.assertTrue(binding.selection.custom)
        self.assertEqual(binding.selection.sample_id, "local")
        scores = runner({"pixels": np.zeros(96, dtype=np.uint8)})["scores"]
        np.testing.assert_array_equal(scores, [[1, 2, 3]])

    def test_invalid_contract_or_missing_file_fails_before_sdk(self):
        from utils.py_utils.model_runner import RuntimeModelRunner
        from unittest.mock import patch

        with patch("utils.py_utils.runtime._default_runtime_factory") as sdk:
            for extra in ({"input_size": (7, 8)}, {"class_count": 0},
                          {"target": "unknown"}, {"score_policy": "guess"}):
                options = dict(target="x5", input_size=(8, 8), class_count=3)
                options.update(extra)
                with self.subTest(extra=extra), self.assertRaises(ValueError):
                    RuntimeModelRunner.from_file("unused.bin", **options)
            with tempfile.TemporaryDirectory() as directory:
                with self.assertRaises(FileNotFoundError):
                    RuntimeModelRunner.from_file(Path(directory) / "absent.bin", target="x5",
                                                 input_size=(8, 8), class_count=3)
            sdk.assert_not_called()


if __name__ == "__main__":
    unittest.main()
