"""Behavioral tests for the readable ResNet classifier entry (classify.py).

These tests pin the user-facing contract of ``ResNetClassifier``: the
three-step flow stays public and consistent with ``predict``, image paths
and BGR arrays are both accepted, input arrays are never modified in
place, per-call geometry never leaks between images of different sizes,
custom class counts work without official manifest registration, and
mismatched labels fail with a concrete error. All runners are injected
host fixtures; no board SDK is loaded and no board inference is claimed.
"""

from __future__ import annotations

from samples.vision.resnet.runtime.python.cli import BINDING_TABLE

import tempfile
import unittest
from pathlib import Path

import numpy as np

from testsupport import runtime_metadata

_ROOT = Path(__file__).resolve().parents[4]
_WOLF = _ROOT / "samples" / "vision" / "resnet" / "test_data" / "white_wolf.JPEG"


def _fake_runtime(protocol: str, score_sequence, calls: list):
    """One SDK-like object returning queued score tensors per call."""
    metadata = runtime_metadata(protocol)
    output_name = metadata.output_names[0]
    model_name = metadata.model_name
    queue = list(score_sequence)

    class _Runtime:
        def __init__(self):
            self.model_names = [model_name]
            self.input_names = {model_name: list(metadata.input_names)}
            self.input_shapes = {model_name: dict(metadata.input_shapes)}
            self.input_dtypes = {model_name: dict(metadata.input_dtypes)}
            self.output_names = {model_name: list(metadata.output_names)}
            self.output_shapes = {model_name: dict(metadata.output_shapes)}
            self.output_dtypes = {model_name: dict(metadata.output_dtypes)}

        def run(self, payload):
            calls.append(payload)
            return {model_name: {output_name: queue.pop(0)}}

    return _Runtime()


def _score_vector(peak: int, size: int = 1000) -> np.ndarray:
    scores = np.zeros((1, size), dtype=np.float32)
    scores[0, peak] = 8.0
    return scores


def _official_classifier(protocol: str, score_sequence, calls: list, **kwargs):
    from samples.vision.resnet.runtime.python.classify import ResNetClassifier
    from samples.vision.resnet.runtime.python.cli import resolve_selection
    from utils.py_utils.model_runner import RuntimeModelRunner

    selection = resolve_selection(protocol)
    runner = RuntimeModelRunner(
        selection, table=BINDING_TABLE, runtime=_fake_runtime(protocol, score_sequence, calls))
    return ResNetClassifier(selection.model_path, target=protocol, runner=runner, **kwargs)


def _custom_classifier(class_count: int, scores: np.ndarray, calls: list, **kwargs):
    from samples.vision.resnet.runtime.python.classify import ResNetClassifier
    from utils.py_utils.model_runner import RuntimeModelRunner
    from unittest.mock import patch

    class _Runtime:
        def __init__(self):
            self.model_names = ["custom_resnet"]
            self.input_names = {"custom_resnet": ["data"]}
            self.input_shapes = {"custom_resnet": {"data": (1, 3, 64, 64)}}
            self.input_dtypes = {"custom_resnet": {"data": "U8"}}
            self.output_names = {"custom_resnet": ["scores"]}
            self.output_shapes = {"custom_resnet": {"scores": (1, class_count)}}
            self.output_dtypes = {"custom_resnet": {"scores": "F32"}}

        def run(self, payload):
            calls.append(payload)
            return {"custom_resnet": {"scores": scores}}

    with patch("utils.py_utils.runtime._default_runtime_factory", return_value=lambda path: _Runtime()), \
            patch("utils.py_utils.platforms.require_execution_target"):
        # The constructor checks a real path before loading the injected SDK.
        with tempfile.NamedTemporaryFile(suffix=".bin") as artifact:
            model = ResNetClassifier(artifact.name, target="x5", input_size=(64, 64),
                                     class_count=class_count, **kwargs)
    return model, model.binding.selection



class ResNetClassifierFlowTests(unittest.TestCase):
    def test_predict_equals_the_explicit_three_steps(self):
        calls: list = []
        scores = _score_vector(42)
        model = _official_classifier("x5", [scores, scores], calls)
        image = np.full((60, 80, 3), 127, dtype=np.uint8)

        via_predict = model.predict(image)
        prepared = model.preprocess(image)
        outputs = model.infer(prepared)
        via_explicit = model.postprocess(outputs)

        self.assertEqual(
            via_predict.class_ids.tolist(), via_explicit.class_ids.tolist())
        np.testing.assert_array_equal(via_predict.scores, via_explicit.scores)
        self.assertEqual(via_predict.labels, via_explicit.labels)

    def test_predict_calls_the_runner_once_per_image(self):
        calls: list = []
        model = _official_classifier(
            "x5", [_score_vector(1), _score_vector(2)], calls)

        model.predict(np.full((30, 40, 3), 90, dtype=np.uint8))
        model.predict(np.full((30, 40, 3), 90, dtype=np.uint8))

        self.assertEqual(len(calls), 2)

    def test_input_array_is_not_modified_in_place(self):
        calls: list = []
        model = _official_classifier("x5", [_score_vector(5)], calls)
        image = np.arange(60 * 80 * 3, dtype=np.uint8).reshape(60, 80, 3)
        before = image.copy()

        model.predict(image)

        np.testing.assert_array_equal(image, before)

    def test_different_size_images_do_not_share_results_or_context(self):
        calls: list = []
        model = _official_classifier(
            "s100", [_score_vector(7), _score_vector(9)], calls)
        wide = np.full((60, 80, 3), 30, dtype=np.uint8)
        small = np.full((20, 20, 3), 200, dtype=np.uint8)

        first = model.predict(wide)
        second = model.predict(small)

        # Each call consumed exactly its own queued output: the peak class
        # of call one is 7, of call two is 9 — no stale score reuse.
        self.assertEqual(int(first.class_ids[0]), 7)
        self.assertEqual(int(second.class_ids[0]), 9)
        self.assertEqual(len(calls), 2)

    def test_prepared_transform_is_per_call_not_stale(self):
        calls: list = []
        model = _official_classifier("s100", [_score_vector(1)] * 3, calls)
        wide = np.full((60, 80, 3), 30, dtype=np.uint8)
        small = np.full((20, 20, 3), 200, dtype=np.uint8)

        first = model.preprocess(wide)
        second = model.preprocess(small)
        third = model.preprocess(wide)

        self.assertEqual(
            (first.transform.original_height, first.transform.original_width),
            (60, 80))
        self.assertEqual(
            (second.transform.original_height, second.transform.original_width),
            (20, 20))
        self.assertEqual(first.transform, third.transform)
        self.assertNotEqual(first.transform, second.transform)


class ResNetClassifierSourceTests(unittest.TestCase):
    def test_predict_accepts_a_local_image_path(self):
        import cv2

        calls: list = []
        scores = _score_vector(11)
        model = _official_classifier("x5", [scores] * 3, calls)
        expected = model.predict(cv2.imread(str(_WOLF), cv2.IMREAD_COLOR))

        from_string = model.predict(str(_WOLF))
        from_path = model.predict(_WOLF)

        self.assertEqual(
            from_string.class_ids.tolist(), expected.class_ids.tolist())
        np.testing.assert_array_equal(from_string.scores, expected.scores)
        self.assertEqual(
            from_path.class_ids.tolist(), expected.class_ids.tolist())

    def test_missing_image_path_error_names_the_path(self):
        calls: list = []
        model = _official_classifier("x5", [_score_vector(1)], calls)
        missing = "/nonexistent/dir/image.jpg"

        with self.assertRaises(FileNotFoundError) as raised:
            model.predict(missing)

        self.assertIn(missing, str(raised.exception))

    def test_unsupported_source_type_is_a_concrete_type_error(self):
        calls: list = []
        model = _official_classifier("x5", [_score_vector(1)], calls)

        with self.assertRaises(TypeError):
            model.predict(12345)


class ResNetClassifierCustomModelTests(unittest.TestCase):
    def test_explicit_probability_output_is_not_softmaxed(self):
        scores = np.array([[0.1, 0.6, 0.2, 0.1]], dtype=np.float32)
        model, _ = _custom_classifier(4, scores, [], top_k=2, score_policy="none")
        result = model.predict(np.zeros((32, 48, 3), dtype=np.uint8))
        self.assertEqual(result.class_ids.tolist(), [1, 2])
        np.testing.assert_array_equal(result.scores, scores[0, [1, 2]])

    def test_invalid_model_parameters_fail_before_runtime_loading(self):
        from unittest.mock import patch
        from samples.vision.resnet.runtime.python.classify import ResNetClassifier

        for options in ({"input_size": (63, 64)}, {"class_count": 0},
                        {"resize_type": 2}, {"score_policy": "guess"}):
            with self.subTest(options=options), patch("utils.py_utils.runtime._default_runtime_factory") as sdk:
                with self.assertRaises(ValueError):
                    ResNetClassifier("unused.bin", target="x5", **options)
                sdk.assert_not_called()

    def test_custom_class_count_returns_ids_without_official_labels(self):
        calls: list = []
        scores = np.zeros((1, 4), dtype=np.float32)
        scores[0, 2] = 6.0
        model, selection = _custom_classifier(4, scores, calls, top_k=2)

        result = model.predict(np.full((32, 48, 3), 60, dtype=np.uint8))

        # A 4-class custom output yields class IDs 2 and the runner-up —
        # never an ImageNet label and never an official asset requirement.
        self.assertEqual(int(result.class_ids[0]), 2)
        self.assertLessEqual(int(result.class_ids[1]), 3)
        self.assertEqual(result.labels, tuple(str(int(i)) for i in result.class_ids))
        self.assertEqual(selection.variant, "custom")

    def test_label_length_mismatch_fails_with_concrete_counts(self):
        calls: list = []
        scores = np.zeros((1, 4), dtype=np.float32)

        with self.assertRaises(ValueError) as raised:
            _custom_classifier(4, scores, calls, top_k=2, labels=["cat", "dog"])

        message = str(raised.exception)
        self.assertIn("2", message)
        self.assertIn("4", message)

    def test_custom_output_shape_mismatch_still_fails_binding(self):
        from utils.py_utils.cls_binding import RuntimeMetadata, bind_model
        _, selection = _custom_classifier(4, np.zeros((1, 4), dtype=np.float32), [], top_k=2)
        wrong = RuntimeMetadata.from_mapping(
            {
                "model_name": "custom_resnet",
                "input_names": ["data"],
                "input_shapes": {"data": (1, 3, 64, 64)},
                "input_dtypes": {"data": "U8"},
                "output_names": ["scores"],
                "output_shapes": {"scores": (1, 1000)},
                "output_dtypes": {"scores": "F32"},
            }
        )

        # The declared 4-class contract still validates the actual tensor:
        # a 1000-class output is a mismatch, not a silent reinterpretation.
        with self.assertRaises(Exception) as raised:
            bind_model(BINDING_TABLE, selection, wrong)
        self.assertIn("1000", str(raised.exception))

    def test_official_path_still_requires_asset_id_for_model_path(self):
        from utils.py_utils.cls_binding import UnsupportedAssetError
        from samples.vision.resnet.runtime.python.cli import resolve_selection

        with self.assertRaises(UnsupportedAssetError):
            resolve_selection("x5", model_path="/tmp/lookalike_resnet18.bin")


class ResNetClassifierLabelTests(unittest.TestCase):
    def test_matching_sequence_labels_resolve_names(self):
        calls: list = []
        scores = _score_vector(3)
        model = _official_classifier(
            "x5", [scores], calls,
            top_k=1, labels=[f"class_{i}" for i in range(1000)])

        result = model.predict(np.full((28, 34, 3), 60, dtype=np.uint8))

        self.assertEqual(result.labels, ("class_3",))

    def test_mapping_labels_covering_the_top_classes_resolve(self):
        calls: list = []
        scores = _score_vector(3)
        model = _official_classifier(
            "x5", [scores], calls, top_k=1, labels={3: "wolf"})

        result = model.predict(np.full((28, 34, 3), 60, dtype=np.uint8))

        self.assertEqual(result.labels, ("wolf",))

    def test_mapping_labels_with_invalid_keys_fail_clearly(self):
        calls: list = []

        with self.assertRaises(ValueError) as raised:
            _official_classifier(
                "x5", [_score_vector(1)], calls, labels={1000: "overflow"})

        self.assertIn("1000", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
