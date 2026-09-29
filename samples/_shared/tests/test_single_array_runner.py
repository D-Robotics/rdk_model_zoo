# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""One-array transport boundaries, independent of point/image task numerics."""

from pathlib import Path
from types import SimpleNamespace
import unittest
import numpy as np


class SingleArrayRunnerTests(unittest.TestCase):
    def runner(self, **kwargs):
        from samples._shared.single_array_runner import SingleArrayRunner

        runtime = SimpleNamespace(
            model_names=["model"],
            input_names={"model": ["in"]},
            input_shapes={"model": {"in": (1, 3, 2, 2)}},
            input_dtypes={"model": {"in": "NV12"}},
            output_names={"model": ["out"]},
            output_shapes={"model": {"out": (1, 2)}},
            output_dtypes={"model": {"out": "float32"}},
        )
        self.raw = np.array([[-3.0, 7.0]], np.float32)
        self.calls = []
        runtime.run = lambda tensors: self.calls.append(tensors) or {
            "model": {"out": self.raw}
        }
        runtime.set_scheduling_params = lambda **kwargs: self.calls.append(kwargs)
        selection = SimpleNamespace(
            target="x5", asset=None, model_path=Path("/not-loaded.bin")
        )

        def bind(selection, metadata):
            return SimpleNamespace(
                model_name="model",
                input_name="in",
                output_name="out",
                metadata=metadata,
            )

        options = dict(
            binding_loader=bind,
            physical_input=lambda binding: ((3, 2), np.uint8),
            task_name="fixture",
            runtime=runtime,
        )
        options.update(kwargs)
        return SingleArrayRunner(selection, **options)

    def test_physical_input_differs_from_logical_metadata(self):
        runner = self.runner()
        result = runner({"in": np.ones((3, 2), np.uint8)})
        self.assertEqual(self.calls[0]["model"]["in"].shape, (3, 2))
        np.testing.assert_array_equal(result, [[-3, 7]])
        self.raw[:] = 9
        np.testing.assert_array_equal(result, [[-3, 7]])

    def test_invalid_inputs_never_reach_runtime(self):
        runner = self.runner()
        for tensors in (
            {"wrong": np.ones((3, 2), np.uint8)},
            {"in": np.ones((1, 3, 2, 2), np.uint8)},
            {"in": np.ones((3, 2), np.float32)},
        ):
            with self.assertRaises(ValueError):
                runner(tensors)
        self.assertEqual(self.calls, [])

    def test_bad_outputs_rejected_without_decoding(self):
        for result in (
            {"other": {"out": np.ones((1, 2), np.float32)}},
            {"model": {"wrong": np.ones((1, 2), np.float32)}},
            {"model": {"out": np.ones((1, 2), np.int32)}},
            {"model": {"out": np.full((1, 2), np.nan, np.float32)}},
        ):
            runner = self.runner()
            runner.runtime.run = lambda tensors: result
            with self.assertRaises(ValueError):
                runner({"in": np.ones((3, 2), np.uint8)})

    def test_binding_failure_discards_runtime(self):
        def fail(*args):
            raise ValueError("bad metadata")

        runner = self.runner(binding_loader=fail)
        with self.assertRaisesRegex(ValueError, "bad metadata"):
            runner.load()
        self.assertFalse(runner.loaded)
        with self.assertRaises(RuntimeError):
            _ = runner.runtime

    def test_scheduling_none_is_noop_and_values_are_scoped(self):
        runner = self.runner()
        runner.set_scheduling_params()
        self.assertFalse(runner.loaded)
        runner.set_scheduling_params(priority=3, bpu_cores=[0, 1])
        self.assertEqual(
            self.calls, [{"priority": {"model": 3}, "bpu_cores": {"model": [0, 1]}}]
        )
        for params in ({"priority": 256}, {"bpu_cores": []}, {"bpu_cores": [-1]}):
            with self.assertRaises(ValueError):
                runner.set_scheduling_params(**params)

    def test_real_loading_checks_identity_and_artifact_before_sdk(self):
        from unittest.mock import patch

        order = []
        runner = self.runner(
            runtime=None, execution_target_gate=lambda target: order.append("identity")
        )

        def reject(*args):
            order.append("artifact")
            raise ValueError("checksum mismatch")

        with patch(
            "samples._shared.single_array_runner.verify_asset_file", side_effect=reject
        ), patch(
            "samples._shared.single_array_runner._default_runtime_factory"
        ) as factory:
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                runner.load()
            factory.assert_not_called()
        self.assertEqual(order, ["identity", "artifact"])
        self.assertFalse(runner.loaded)


class SplitInputArrayRunnerTests(unittest.TestCase):
    def test_two_named_physical_inputs_feed_one_raw_array(self):
        from samples._shared.single_array_runner import SingleArrayRunner

        selection = SimpleNamespace(
            target="s100", asset=None, model_path=Path("/not-loaded.hbm")
        )
        runtime = SimpleNamespace(
            model_names=["model"],
            input_names={"model": ["y", "uv"]},
            input_shapes={"model": {"y": (1, 4, 8, 1), "uv": (1, 2, 4, 2)}},
            input_dtypes={"model": {"y": "uint8", "uv": "uint8"}},
            output_names={"model": ["scores"]},
            output_shapes={"model": {"scores": (1, 2)}},
            output_dtypes={"model": {"scores": "int32"}},
        )
        calls = []
        raw = np.array([[5, 7]], np.int32)
        runtime.run = lambda tensors: calls.append(tensors) or {
            "model": {"scores": raw}
        }

        def bind(selection, metadata):
            return SimpleNamespace(
                model_name="model", output_name="scores", metadata=metadata
            )

        runner = SingleArrayRunner(
            selection,
            binding_loader=bind,
            task_name="split",
            runtime=runtime,
            physical_inputs=lambda binding: {
                "y": ((1, 4, 8, 1), "uint8"),
                "uv": ((1, 2, 4, 2), "uint8"),
            },
        )
        inputs = {
            "y": np.zeros((1, 4, 8, 1), np.uint8),
            "uv": np.zeros((1, 2, 4, 2), np.uint8),
        }
        actual = runner(inputs)
        self.assertEqual(set(calls[0]["model"]), {"y", "uv"})
        raw[:] = 0
        np.testing.assert_array_equal(actual, [[5, 7]])
        for bad in (
            {"y": inputs["y"]},
            {**inputs, "extra": inputs["y"]},
            {**inputs, "uv": np.zeros((1, 2, 4, 2), np.float32)},
        ):
            with self.assertRaises(ValueError):
                runner(bad)
        self.assertEqual(len(calls), 1)


class NamedArrayRunnerTests(unittest.TestCase):
    """Multiple raw outputs, including unconsumed auxiliaries, stay named."""

    def make_runner(self):
        from samples._shared.single_array_runner import NamedArrayRunner

        self.raw = {
            "embedding": np.array([[-0.5, 1.2]], np.float32),
            "labels": np.array([[0, 1]], np.int64),
            "aux": np.array([[4]], np.int32),
        }
        runtime = SimpleNamespace(
            model_names=["m"],
            input_names={"m": ["in"]},
            input_shapes={"m": {"in": (1, 2)}},
            input_dtypes={"m": {"in": "float32"}},
            output_names={"m": list(self.raw)},
            output_shapes={"m": {k: v.shape for k, v in self.raw.items()}},
            output_dtypes={"m": {k: str(v.dtype) for k, v in self.raw.items()}},
        )
        self.calls = []
        runtime.run = lambda inputs: self.calls.append(inputs) or {
            "m": dict(reversed(list(self.raw.items())))
        }
        selection = SimpleNamespace(
            target="s100", asset=None, model_path=Path("/fixture")
        )

        def bind(selection, metadata):
            return SimpleNamespace(model_name="m", input_name="in", metadata=metadata)

        return NamedArrayRunner(
            selection,
            binding_loader=bind,
            physical_input=lambda b: ((1, 2), "float32"),
            task_name="named",
            runtime=runtime,
        )

    def test_returns_owned_raw_arrays_without_order_or_decode_assumptions(self):
        runner = self.make_runner()
        result = runner({"in": np.zeros((1, 2), np.float32)})
        self.assertEqual(set(result), set(self.raw))
        for name in result:
            np.testing.assert_array_equal(result[name], self.raw[name])
            self.assertFalse(np.shares_memory(result[name], self.raw[name]))
        self.assertEqual(result["labels"].dtype, np.int64)
        self.assertEqual(result["embedding"][0, 0], -0.5)

    def test_rejects_missing_extra_wrong_type_shape_and_nonfinite_auxiliary(self):
        runner = self.make_runner()
        for bad in (
            {k: v for k, v in self.raw.items() if k != "aux"},
            {**self.raw, "extra": np.zeros((1, 1), np.float32)},
            {**self.raw, "labels": np.array([[0, 1]], np.int32)},
            {**self.raw, "aux": np.zeros((2,), np.int32)},
            {**self.raw, "embedding": np.full((1, 2), np.nan, np.float32)},
        ):
            runner.runtime.run = lambda inputs: {"m": bad}
            with self.assertRaises(ValueError):
                runner({"in": np.zeros((1, 2), np.float32)})
