"""Behavioral tests for the readable FasterNet classifier entry (classify.py).

These tests pin the user-facing contract of ``FasterNetClassifier``: the three-step
flow stays public and consistent with ``predict`` on every published
protocol, image paths and BGR arrays are both accepted, input arrays are
never modified in place, per-call geometry never leaks between images of
different sizes, the established ``pre_process``/``forward``/``post_process``
spellings delegate to the same implementations, and mismatched labels or
score shapes fail with concrete errors.  All runners are injected host
fixtures; no board SDK is loaded and no board inference is claimed.
"""

from __future__ import annotations

from samples.vision.fasternet.runtime.python.model_binding import BINDING_TABLE

from pathlib import Path
import unittest
from unittest import mock

import numpy as np

from samples.vision.fasternet.runtime.python.model_binding import (
    BindingError,
    RuntimeMetadata,
    resolve_selection,
)

_ROOT = Path(__file__).resolve().parents[4]
_IMAGE = _ROOT / "samples" / "vision" / "fasternet" / "test_data" / "drake.JPEG"
_TARGETS = ("x5",)


def _runtime_metadata(selection):
    """Synthetic host metadata matching the bound contract (protocol-aware)."""

    contract = selection.contract
    height, width = contract.input_height, contract.input_width
    if contract.input_protocol == "packed_nv12":
        return RuntimeMetadata.from_mapping(
            {
                "model_name": "fasternet",
                "input_names": ["data"],
                "input_shapes": {"data": (1, 3, height, width)},
                "input_dtypes": {"data": "U8"},
                "output_names": ["prob"],
                "output_shapes": {"prob": (1, contract.class_count, 1, 1)},
                "output_dtypes": {"prob": "F32"},
            }
        )
    return RuntimeMetadata.from_mapping(
        {
            "model_name": "fasternet",
            "input_names": ["input_y", "input_uv"],
            "input_shapes": {
                "input_y": (1, height, width, 1),
                "input_uv": (1, height // 2, width // 2, 2),
            },
            "input_dtypes": {"input_y": "U8", "input_uv": "U8"},
            "output_names": ["output"],
            "output_shapes": {"output": (1, contract.class_count)},
            "output_dtypes": {"output": "F32"},
        }
    )


def _shaped_scores(selection, peak):
    """One score vector peaked at ``peak`` in the protocol's published shape."""

    scores = np.zeros((1, selection.contract.class_count), dtype=np.float32)
    scores[0, peak % selection.contract.class_count] = 8.0
    if selection.contract.input_protocol == "packed_nv12":
        classes = selection.contract.class_count
        return scores.reshape(1, classes, 1, 1)
    return scores


def _fake_runtime(selection, score_sequence, calls):
    """One SDK-like object returning queued score tensors per call."""

    metadata = _runtime_metadata(selection)
    model_name = metadata.model_name
    output_name = metadata.output_names[0]
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


def _classifier(target, score_sequence, calls, **kwargs):
    from samples.vision.fasternet.runtime.python.classify import FasterNetClassifier
    from utils.py_utils.model_runner import RuntimeModelRunner

    selection = resolve_selection(target)
    runner = RuntimeModelRunner(
        selection, runtime=_fake_runtime(selection, score_sequence, calls), table=BINDING_TABLE)
    return FasterNetClassifier(selection, runner=runner, **kwargs)


def _class_count(target):
    return resolve_selection(target).contract.class_count


class ClassifierFlowTests(unittest.TestCase):
    def test_predict_equals_the_explicit_three_steps_on_each_protocol(self):
        for target in _TARGETS:
            with self.subTest(target=target):
                calls: list = []
                peak = 7
                scores = _shaped_scores(resolve_selection(target), peak)
                model = _classifier(target, [scores, scores], calls)
                image = np.full((60, 80, 3), 127, dtype=np.uint8)

                via_predict = model.predict(image)
                prepared = model.preprocess(image)
                outputs = model.infer(prepared)
                via_explicit = model.postprocess(outputs)

                self.assertEqual(
                    via_predict.class_ids.tolist(), via_explicit.class_ids.tolist())
                np.testing.assert_array_equal(via_predict.scores, via_explicit.scores)
                self.assertEqual(via_predict.labels, via_explicit.labels)
                self.assertEqual(int(via_predict.class_ids[0]), peak % _class_count(target))

    def test_legacy_stage_names_delegate_to_the_same_flow(self):
        calls: list = []
        target = _TARGETS[0]
        scores = _shaped_scores(resolve_selection(target), 3)
        model = _classifier(target, [scores, scores], calls)
        image = np.full((30, 40, 3), 90, dtype=np.uint8)

        via_predict = model.predict(image)
        prepared = model.pre_process(image)
        outputs = model.forward(prepared)
        via_legacy = model.post_process(outputs)

        self.assertEqual(
            via_predict.class_ids.tolist(), via_legacy.class_ids.tolist())
        np.testing.assert_array_equal(via_predict.scores, via_legacy.scores)
        self.assertEqual(via_predict.labels, via_legacy.labels)

    def test_forward_delegates_to_infer_for_prepared_and_raw_tensors(self):
        calls: list = []
        target = _TARGETS[0]
        scores = _shaped_scores(resolve_selection(target), 1)
        model = _classifier(target, [scores] * 2, calls)
        prepared = model.preprocess(np.full((30, 40, 3), 90, dtype=np.uint8))
        sentinel = object()

        with mock.patch.object(model, "infer",
                               return_value=sentinel) as patched_infer:
            via_prepared = model.forward(prepared)
            via_raw = model.forward(prepared.tensors)

        # forward must delegate to the same infer implementation for both
        # legal input forms; it never runs the runner itself.
        forwarded = [call.args[0] for call in patched_infer.call_args_list]
        self.assertIs(via_prepared, sentinel)
        self.assertIs(via_raw, sentinel)
        self.assertEqual(forwarded, [prepared, prepared.tensors])
        self.assertEqual(len(calls), 0)

    def test_predict_calls_the_runner_once_per_image(self):
        calls: list = []
        target = _TARGETS[0]
        model = _classifier(
            target,
            [_shaped_scores(resolve_selection(target), 1),
             _shaped_scores(resolve_selection(target), 2)],
            calls)

        model.predict(np.full((30, 40, 3), 90, dtype=np.uint8))
        model.predict(np.full((30, 40, 3), 90, dtype=np.uint8))

        self.assertEqual(len(calls), 2)

    def test_input_array_is_not_modified_in_place(self):
        calls: list = []
        target = _TARGETS[0]
        model = _classifier(
            target, [_shaped_scores(resolve_selection(target), 5)], calls)
        image = np.arange(60 * 80 * 3, dtype=np.uint8).reshape(60, 80, 3)
        before = image.copy()

        model.predict(image)

        np.testing.assert_array_equal(image, before)

    def test_different_size_images_do_not_share_results_or_context(self):
        calls: list = []
        target = _TARGETS[-1]
        classes = _class_count(target)
        model = _classifier(
            target,
            [_shaped_scores(resolve_selection(target), 7),
             _shaped_scores(resolve_selection(target), 9)],
            calls)
        wide = np.full((60, 80, 3), 30, dtype=np.uint8)
        small = np.full((20, 20, 3), 200, dtype=np.uint8)

        first = model.predict(wide)
        second = model.predict(small)

        # Each call consumed exactly its own queued output: no stale reuse.
        self.assertEqual(int(first.class_ids[0]), 7 % classes)
        self.assertEqual(int(second.class_ids[0]), 9 % classes)
        self.assertEqual(len(calls), 2)

    def test_prepared_transform_is_per_call_not_stale(self):
        calls: list = []
        target = _TARGETS[-1]
        scores = _shaped_scores(resolve_selection(target), 1)
        model = _classifier(target, [scores] * 3, calls)
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

    def test_output_shape_mismatch_propagates_through_predict(self):
        calls: list = []
        target = _TARGETS[0]
        selection = resolve_selection(target)
        wrong = np.zeros((1, selection.contract.class_count + 5), dtype=np.float32)
        metadata = _runtime_metadata(selection)
        model_name = metadata.model_name
        output_name = metadata.output_names[0]

        class _WrongRuntime:
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
                return {model_name: {output_name: wrong}}

        from samples.vision.fasternet.runtime.python.classify import FasterNetClassifier
        from utils.py_utils.model_runner import RuntimeModelRunner

        runner = RuntimeModelRunner(selection, runtime=_WrongRuntime(), table=BINDING_TABLE)
        model = FasterNetClassifier(selection, runner=runner)

        with self.assertRaises(BindingError):
            model.predict(np.full((20, 20, 3), 60, dtype=np.uint8))
        # The failure happened at execution time, after exactly one call.
        self.assertEqual(len(calls), 1)


class ClassifierSourceTests(unittest.TestCase):
    def test_predict_accepts_a_local_image_path(self):
        import cv2

        calls: list = []
        target = _TARGETS[0]
        scores = _shaped_scores(resolve_selection(target), 11)
        model = _classifier(target, [scores] * 3, calls)
        expected = model.predict(cv2.imread(str(_IMAGE), cv2.IMREAD_COLOR))

        from_string = model.predict(str(_IMAGE))
        from_path = model.predict(_IMAGE)

        self.assertEqual(
            from_string.class_ids.tolist(), expected.class_ids.tolist())
        np.testing.assert_array_equal(from_string.scores, expected.scores)
        self.assertEqual(
            from_path.class_ids.tolist(), expected.class_ids.tolist())

    def test_missing_image_path_error_names_the_path(self):
        calls: list = []
        target = _TARGETS[0]
        model = _classifier(
            target, [_shaped_scores(resolve_selection(target), 1)], calls)
        missing = "/nonexistent/dir/image.jpg"

        with self.assertRaises(FileNotFoundError) as raised:
            model.predict(missing)

        self.assertIn(missing, str(raised.exception))

    def test_unsupported_source_type_is_a_concrete_type_error(self):
        calls: list = []
        target = _TARGETS[0]
        model = _classifier(
            target, [_shaped_scores(resolve_selection(target), 1)], calls)

        with self.assertRaises(TypeError):
            model.predict(12345)


class ClassifierLabelTests(unittest.TestCase):
    def test_matching_sequence_labels_resolve_names(self):
        calls: list = []
        target = _TARGETS[0]
        peak = 3
        scores = _shaped_scores(resolve_selection(target), peak)
        model = _classifier(
            target, [scores], calls,
            top_k=1, labels=[f"class_{i}" for i in range(_class_count(target))])

        result = model.predict(np.full((28, 34, 3), 60, dtype=np.uint8))

        self.assertEqual(result.labels, (f"class_{peak % _class_count(target)}",))

    def test_mapping_labels_covering_the_top_classes_resolve(self):
        calls: list = []
        target = _TARGETS[0]
        peak = 3
        scores = _shaped_scores(resolve_selection(target), peak)
        model = _classifier(target, [scores], calls, top_k=1,
                            labels={peak % _class_count(target): "target"})

        result = model.predict(np.full((28, 34, 3), 60, dtype=np.uint8))

        self.assertEqual(result.labels, ("target",))

    def test_mapping_labels_with_invalid_keys_fail_clearly(self):
        calls: list = []
        target = _TARGETS[0]
        classes = _class_count(target)

        with self.assertRaises(ValueError) as raised:
            _classifier(
                target,
                [_shaped_scores(resolve_selection(target), 1)], calls,
                labels={classes: "overflow"})

        self.assertIn(str(classes), str(raised.exception))

    def test_sequence_labels_of_wrong_length_fail_with_concrete_counts(self):
        calls: list = []
        target = _TARGETS[0]
        classes = _class_count(target)

        with self.assertRaises(ValueError) as raised:
            _classifier(
                target,
                [_shaped_scores(resolve_selection(target), 1)], calls,
                labels=["cat", "dog"])

        message = str(raised.exception)
        self.assertIn("2", message)
        self.assertIn(str(classes), message)

    def test_top_k_outside_the_class_count_fails(self):
        calls: list = []
        target = _TARGETS[0]
        classes = _class_count(target)

        for top_k in (0, classes + 1):
            with self.subTest(top_k=top_k):
                with self.assertRaises(ValueError):
                    _classifier(
                        target,
                        [_shaped_scores(resolve_selection(target), 1)], calls,
                        top_k=top_k)


if __name__ == "__main__":
    unittest.main()
