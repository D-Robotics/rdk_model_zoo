# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0

"""Board-output dict-order contract for FCOS (X5 evidence 2026-09-24).

The board ``hbm_runtime`` ``run()`` returns its fifteen outputs in a dict
whose key order differs from ``metadata.output_names`` while the name set
is identical.  These tests pin the remediated ``validate_outputs``: exact
name-set matching with per-binding shape/dtype/finiteness checks, caller
mapping identity preserved, and every strict rejection retained.
"""

from __future__ import annotations

import importlib
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


BINDING = "samples.vision.fcos.runtime.python.model_binding"
TASK = "samples.vision.fcos.runtime.python.fcos"
RUNNER_MODULE = "samples.vision.fcos.runtime.python.model_runner"
COMPARE = "samples.vision.fcos.evaluator.compare"
SAMPLE = Path(__file__).resolve().parents[1]
STRIDES = (8, 16, 32, 64, 128)


class QuantType:
    def __init__(self, name="SCALE"):
        self.name = name


class Quant:
    """Minimal runtime-like quantization descriptor."""

    def __init__(self, scale=0.1, zero_point=0.0, axis=0, quant_type="SCALE"):
        self.quant_type = QuantType(quant_type)
        self.scale = np.asarray(scale, dtype=np.float32)
        self.zero_point = np.asarray(zero_point, dtype=np.float32)
        self.axis = axis


def _shapes(size=512):
    return {
        **{f"cls_{s}": (1, size // s, size // s, 80) for s in STRIDES},
        **{f"box_{s}": (1, size // s, size // s, 4) for s in STRIDES},
        **{f"center_{s}": (1, size // s, size // s, 1) for s in STRIDES},
    }


def metadata_for(dtype="int8"):
    shapes = _shapes()
    names = tuple(shapes)
    return {
        "model_name": "fcos_efficientnetb0",
        "input_names": ["input"],
        "input_shapes": {"input": (1, 3, 512, 512)},
        "input_dtypes": {"input": "NV12"},
        "output_names": names,
        "output_shapes": shapes,
        "output_dtypes": {name: dtype for name in names},
        "output_quants": {name: Quant() for name in names},
    }


def fixture_outputs(size=512):
    shapes = _shapes(size)
    outputs = {name: np.full(shape, -12, dtype=np.int8) for name, shape in shapes.items()}
    for index, stride in enumerate(STRIDES):
        outputs[f"cls_{stride}"][0, 0, 0, index] = 10
        outputs[f"center_{stride}"][0, 0, 0, 0] = 10
        outputs[f"box_{stride}"][0, 0, 0, :] = 1
    return outputs


def shuffled(values):
    """The same mapping content in a different insertion order."""

    return dict(reversed(list(values.items())))


class FakeBoardRuntime:
    """Runtime whose run() returns outputs in a shuffled dict order."""

    def __init__(self, metadata, raw):
        self.model_names = [metadata["model_name"]]
        self.input_names = {metadata["model_name"]: metadata["input_names"]}
        self.input_shapes = {metadata["model_name"]: metadata["input_shapes"]}
        self.input_dtypes = {metadata["model_name"]: metadata["input_dtypes"]}
        self.output_names = {metadata["model_name"]: metadata["output_names"]}
        self.output_shapes = {metadata["model_name"]: metadata["output_shapes"]}
        self.output_dtypes = {metadata["model_name"]: metadata["output_dtypes"]}
        self.output_quants = {metadata["model_name"]: metadata["output_quants"]}
        self._raw = raw

    def run(self, inputs):
        return {self.model_names[0]: shuffled(self._raw)}

    def set_scheduling_params(self, **kwargs):
        self.scheduling = kwargs


class ShuffledDictAcceptanceTests(unittest.TestCase):
    def _binding(self, dtype="int8"):
        binding_mod = importlib.import_module(BINDING)
        return binding_mod.bind_model(
            binding_mod.resolve_selection(
                "x5",
                asset_id="x5:fcos:fcos_efficientnetb0_detect_512x512_bayese_nv12.bin",
                model_path=str(self._model_file()),
            ),
            metadata_for(dtype),
        )

    def _model_file(self):
        if not hasattr(self, "_model"):
            handle = tempfile.TemporaryDirectory()
            self.addCleanup(handle.cleanup)
            self._model = Path(handle.name) / "fcos-fixture.bin"
            self._model.write_bytes(b"fixture-fcos-model")
        return self._model

    def test_validate_outputs_matches_name_set_not_dict_order(self):
        binding = self._binding()
        raw = fixture_outputs()
        ordered = dict(zip(binding.output_names, (raw[name] for name in binding.output_names)))
        # The exact caller mapping object and each ndarray identity survive.
        reversed_dict = shuffled(ordered)
        validated = binding.validate_outputs(reversed_dict)
        self.assertIs(validated, reversed_dict)
        for name in binding.output_names:
            self.assertIs(validated[name], raw[name])

    def test_task_predict_accepts_shuffled_runner_outputs(self):
        fcos_mod = importlib.import_module(TASK)
        binding = self._binding()
        raw = fixture_outputs()
        ordered_task = fcos_mod.FCOSTask(lambda tensors: raw, binding)
        shuffled_task = fcos_mod.FCOSTask(lambda tensors: shuffled(raw), binding)
        image = np.zeros((300, 500, 3), dtype=np.uint8)
        from_ordered = ordered_task.predict(image)
        from_shuffled = shuffled_task.predict(image)
        self.assertEqual(from_ordered, from_shuffled)
        self.assertEqual(from_shuffled.boxes.shape, (5, 4))

    def test_runner_returns_shuffled_board_dict_unchanged(self):
        runner_mod = importlib.import_module(RUNNER_MODULE)
        binding_mod = importlib.import_module(BINDING)
        metadata = metadata_for()
        raw = fixture_outputs()
        runner = runner_mod.RuntimeModelRunner(
            binding_mod.resolve_selection(
                "x5",
                asset_id="x5:fcos:fcos_efficientnetb0_detect_512x512_bayese_nv12.bin",
                model_path=str(self._model_file()),
            ),
            runtime=FakeBoardRuntime(metadata, raw),
        )
        binding = runner.load()
        tensors = {"input": np.zeros(512 * 512 * 3 // 2, dtype=np.uint8)}
        outputs = runner(tensors)
        self.assertEqual(set(outputs), set(binding.output_names))
        self.assertIs(outputs["cls_8"], raw["cls_8"])

    def test_evaluator_passes_with_shuffled_runtime_output_order(self):
        compare = importlib.import_module(COMPARE)
        fcos_mod = importlib.import_module(TASK)
        binding_mod = importlib.import_module(BINDING)
        from samples._shared.runtime_meta import RuntimeMetadata

        metadata = metadata_for()
        raw = fixture_outputs()
        image = np.zeros((512, 512, 3), dtype=np.uint8)
        selection = binding_mod.resolve_selection(
            "x5",
            asset_id="x5:fcos:fcos_efficientnetb0_detect_512x512_bayese_nv12.bin",
            model_path=str(self._model_file()),
        )

        def legacy_side(selection, image, **kwargs):
            binding = binding_mod.bind_model(selection, metadata)
            task = fcos_mod.FCOSTask(lambda tensors: raw, binding)
            prepared = task.pre_process(image)
            return {
                "metadata": RuntimeMetadata.from_mapping(metadata),
                "inputs": {key: value.copy() for key, value in prepared.tensors.items()},
                "raw": {key: value.copy() for key, value in raw.items()},
                "result": dict(zip(
                    ("boxes", "scores", "class_ids"),
                    task.post_process(raw, prepared.context).as_tuple(),
                )),
            }

        with tempfile.TemporaryDirectory() as temp:
            with mock.patch.object(compare, "require_execution_target", return_value="x5"):
                summary = compare.run_comparison(
                    selection, image, SAMPLE / "test_data" / "bus.jpg",
                    Path(temp) / "shuffled",
                    runtime_factory=lambda path: FakeBoardRuntime(metadata, raw),
                    legacy_runner_factory=legacy_side,
                )
        self.assertEqual(summary["return_code"], 0, summary.get("errors"))
        self.assertTrue(summary["passed"])
        self.assertTrue(summary["checks"]["raw.names"])


class StrictSemanticsTests(unittest.TestCase):
    def setUp(self):
        binding_mod = importlib.import_module(BINDING)
        self.binding_mod = binding_mod
        self.binding = binding_mod.bind_model(
            binding_mod.resolve_selection(
                "x5",
                asset_id="x5:fcos:fcos_efficientnetb0_detect_512x512_bayese_nv12.bin",
                model_path=str(self._model_file()),
            ),
            metadata_for(),
        )

    def _model_file(self):
        if not hasattr(self, "_model"):
            handle = tempfile.TemporaryDirectory()
            self.addCleanup(handle.cleanup)
            self._model = Path(handle.name) / "fcos-fixture.bin"
            self._model.write_bytes(b"fixture-fcos-model")
        return self._model

    def test_missing_output_is_rejected(self):
        raw = fixture_outputs()
        incomplete = dict(raw)
        del incomplete["cls_8"]
        with self.assertRaisesRegex(self.binding_mod.BindingError, "missing"):
            self.binding.validate_outputs(incomplete)

    def test_extra_output_is_rejected(self):
        raw = dict(fixture_outputs())
        raw["aux"] = np.zeros((1,), dtype=np.int8)
        with self.assertRaisesRegex(self.binding_mod.BindingError, "unexpected"):
            self.binding.validate_outputs(raw)

    def test_empty_mapping_is_rejected(self):
        with self.assertRaises(self.binding_mod.BindingError):
            self.binding.validate_outputs({})

    def test_non_mapping_output_is_rejected(self):
        with self.assertRaises(self.binding_mod.BindingError):
            self.binding.validate_outputs([fixture_outputs()["cls_8"]])

    def test_non_array_value_is_rejected(self):
        raw = fixture_outputs()
        raw["cls_8"] = raw["cls_8"].tolist()
        with self.assertRaises(self.binding_mod.BindingError):
            self.binding.validate_outputs(raw)

    def test_wrong_shape_is_rejected_even_when_order_matches(self):
        raw = fixture_outputs()
        raw["box_16"] = raw["box_16"][:, :-1, :, :]
        with self.assertRaises(self.binding_mod.BindingError):
            self.binding.validate_outputs(raw)

    def test_wrong_dtype_is_rejected(self):
        raw = {name: value.astype(np.float32) for name, value in fixture_outputs().items()}
        with self.assertRaises(self.binding_mod.BindingError):
            self.binding.validate_outputs(raw)

    def test_non_finite_values_are_rejected(self):
        binding = self.binding_mod.bind_model(
            self.binding_mod.resolve_selection(
                "x5",
                asset_id="x5:fcos:fcos_efficientnetb0_detect_512x512_bayese_nv12.bin",
                model_path=str(self._model_file()),
            ),
            metadata_for(dtype="float32"),
        )
        raw = {name: value.astype(np.float32) for name, value in fixture_outputs().items()}
        raw["center_8"][0, 0, 0, 0] = np.nan
        with self.assertRaises(self.binding_mod.BindingError):
            binding.validate_outputs(raw)

    def test_task_predict_rejects_runner_missing_one_output(self):
        fcos_mod = importlib.import_module(TASK)
        raw = fixture_outputs()
        incomplete = dict(raw)
        del incomplete["center_128"]
        task = fcos_mod.FCOSTask(lambda tensors: incomplete, self.binding)
        with self.assertRaises(self.binding_mod.BindingError):
            task.predict(np.zeros((64, 96, 3), dtype=np.uint8))


if __name__ == "__main__":
    unittest.main()
