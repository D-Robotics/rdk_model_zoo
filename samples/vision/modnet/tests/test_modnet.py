# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import cv2
import numpy as np


class FakeRuntime:
    model_names = ["modnet"]
    input_names = {"modnet": ["input"]}
    input_shapes = {"modnet": {"input": (1, 3, 512, 512)}}
    input_dtypes = {"modnet": {"input": "float32"}}
    output_names = {"modnet": ["matte"]}
    output_shapes = {"modnet": {"matte": (1, 1, 512, 512)}}
    input_strides = {"modnet": {}}
    output_strides = {"modnet": {}}
    output_dtypes = {"modnet": {"matte": "float32"}}
    output_quants = {"modnet": {}}

    def __init__(self, matte):
        self.matte = matte
        self.calls = []

    def run(self, tensors):
        self.calls.append(tensors)
        return {"modnet": {"matte": self.matte}}

    def set_scheduling_params(self, **kwargs):
        self.scheduling = kwargs


class MODNetTests(unittest.TestCase):
    def test_geometry_matches_fixed_source_numeric_helper(self):
        source_path = Path(__file__).resolve().parents[4] / "platforms/x5/samples/vision/modnet/runtime/python/modnet.py"
        fake_hbm = types.ModuleType("hbm_runtime")
        utils = types.ModuleType("utils")
        utils_py = types.ModuleType("utils.py_utils")
        utils_inspect = types.ModuleType("utils.py_utils.inspect")
        old = {name: sys.modules.get(name) for name in ("hbm_runtime", "utils", "utils.py_utils", "utils.py_utils.inspect")}
        sys.modules.update({"hbm_runtime": fake_hbm, "utils": utils, "utils.py_utils": utils_py, "utils.py_utils.inspect": utils_inspect})
        try:
            spec = importlib.util.spec_from_file_location("legacy_modnet", source_path)
            module = importlib.util.module_from_spec(spec)
            assert spec.loader is not None
            sys.modules["legacy_modnet"] = module
            spec.loader.exec_module(module)
            image = np.arange(3 * 5 * 3, dtype=np.float32).reshape(3, 5, 3)
            source_padded, sx, sy, sw, sh = module.MODNet._resize_with_padding(image, 512)
            unified_mod = importlib.import_module("samples.vision.modnet.runtime.python.modnet")
            unified_padded, context = unified_mod.resize_with_padding(image, 512)
            self.assertTrue(np.array_equal(source_padded, unified_padded))
            self.assertEqual((sx, sy, sw, sh), (context.pad_x, context.pad_y, context.resized_width, context.resized_height))
        finally:
            for name, value in old.items():
                if value is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = value
            sys.modules.pop("legacy_modnet", None)

    def test_binding_accepts_metadata_and_rejects_shape_or_dtype(self):
        binding = importlib.import_module(
            "samples.vision.modnet.runtime.python.model_binding"
        )
        selection = binding.resolve_selection("x5")
        good = {
            "model_names": ("modnet",),
            "model_name": "modnet",
            "input_names": ("input",),
            "input_shapes": {"input": (1, 3, 512, 512)},
            "input_dtypes": {"input": "float32"},
            "output_names": ("matte",),
            "output_shapes": {"matte": (1, 1, 512, 512)},
            "output_dtypes": {"matte": "float32"},
        }
        self.assertEqual(binding.bind_model(selection, good).output_name, "matte")
        for field, value in (
            ("input_shapes", {"input": (1, 3, 256, 256)}),
            ("output_dtypes", {"matte": "int8"}),
        ):
            bad = dict(good)
            bad[field] = value
            with self.assertRaises(binding.MetadataMismatchError):
                binding.bind_model(selection, bad)

    def test_geometry_context_is_per_call_for_a_b_a(self):
        runtime_mod = importlib.import_module(
            "samples.vision.modnet.runtime.python.model_runner"
        )
        task_mod = importlib.import_module(
            "samples.vision.modnet.runtime.python.modnet"
        )
        binding_mod = importlib.import_module(
            "samples.vision.modnet.runtime.python.model_binding"
        )
        selection = binding_mod.resolve_selection("x5")
        matte = np.ones((1, 1, 512, 512), dtype=np.float32)
        runner = runtime_mod.RuntimeModelRunner(selection, runtime=FakeRuntime(matte))
        binding = runner.load()
        task = task_mod.MODNetTask(runner, binding)
        image_a = np.zeros((80, 160, 3), dtype=np.uint8)
        image_b = np.zeros((240, 100, 3), dtype=np.uint8)
        prepared_a1 = task.pre_process(image_a)
        prepared_b = task.pre_process(image_b)
        prepared_a2 = task.pre_process(image_a)
        self.assertNotEqual(prepared_a1.context, prepared_b.context)
        self.assertEqual(prepared_a1.context, prepared_a2.context)
        self.assertEqual(task.post_process(matte, prepared_a1.context).shape, image_a.shape[:2])
        self.assertEqual(task.post_process(matte, prepared_b.context).shape, image_b.shape[:2])
        self.assertEqual(task.post_process(matte, prepared_a2.context).shape, image_a.shape[:2])

    def test_task_uses_source_normalization_and_owned_raw_result(self):
        runtime_mod = importlib.import_module(
            "samples.vision.modnet.runtime.python.model_runner"
        )
        task_mod = importlib.import_module(
            "samples.vision.modnet.runtime.python.modnet"
        )
        binding_mod = importlib.import_module(
            "samples.vision.modnet.runtime.python.model_binding"
        )
        selection = binding_mod.resolve_selection("x5")
        raw = np.zeros((1, 1, 512, 512), dtype=np.float32)
        runner = runtime_mod.RuntimeModelRunner(selection, runtime=FakeRuntime(raw))
        task = task_mod.MODNetTask(runner, runner.load())
        image = np.array([[[0, 10, 20]]], dtype=np.uint8)
        prepared = task.pre_process(image)
        self.assertEqual(prepared.tensors["input"].dtype, np.float32)
        expected_rgb = np.array([20, 10, 0], dtype=np.float32)
        expected = (expected_rgb - 127.5) / 127.5
        self.assertTrue(np.allclose(prepared.tensors["input"][0, :, 0, 0], expected))
        result = task.forward(prepared.tensors)
        self.assertIsNot(result, raw)
        self.assertTrue(np.array_equal(result, raw))

    def test_manual_external_model_requires_exact_asset_id(self):
        binding = importlib.import_module(
            "samples.vision.modnet.runtime.python.model_binding"
        )
        with self.assertRaises(binding.BindingError):
            binding.resolve_selection("x5", model_path="/tmp/modnet.bin")

    def test_cli_list_and_dry_run_do_not_construct_runtime(self):
        main = importlib.import_module("samples.vision.modnet.runtime.python.main")
        with patch.object(main, "RuntimeModelRunner", side_effect=AssertionError):
            self.assertEqual(main.main(["--list-models"]), 0)
            self.assertEqual(main.main(["--dry-run", "--target", "x5"]), 0)

    def test_dry_run_rejects_scheduling_and_ref_size_the_run_would_refuse(self):
        main = importlib.import_module("samples.vision.modnet.runtime.python.main")
        for args in (
            ["--dry-run", "--target", "x5", "--priority", "300"],
            ["--dry-run", "--target", "x5", "--priority", "-1"],
            ["--dry-run", "--target", "x5", "--bpu-cores", "0", "-1"],
            ["--dry-run", "--target", "x5", "--ref-size", "256"],
        ):
            self.assertEqual(main.main(args), 2, args)
        self.assertEqual(
            main.main(["--dry-run", "--target", "x5", "--priority", "5", "--bpu-cores", "0"]), 0
        )

    def test_manual_preparation_never_attempts_network_download(self):
        module = importlib.import_module("samples.vision.modnet.model.download")
        self.assertEqual(module.main(["--target", "x5", "--asset-id", module.ASSET_ID]), 2)

    def test_real_path_gates_before_sdk_and_the_seam_skips_the_gate(self):
        runner_mod = importlib.import_module(
            "samples.vision.modnet.runtime.python.model_runner"
        )
        binding = importlib.import_module(
            "samples.vision.modnet.runtime.python.model_binding"
        )
        selection = binding.resolve_selection("x5")
        with patch.object(runner_mod, "require_execution_target",
                          side_effect=ValueError("no board identity")) as gate:
            with self.assertRaises(ValueError):
                runner_mod.RuntimeModelRunner(selection).load()
            gate.assert_called_once_with("x5")
        matte = np.zeros((1, 1, 512, 512), dtype=np.float32)
        with patch.object(runner_mod, "require_execution_target",
                          side_effect=AssertionError("injected factory is the host seam")):
            runner = runner_mod.RuntimeModelRunner(
                selection, runtime_factory=lambda path: FakeRuntime(matte)
            )
            self.assertIsNotNone(runner.load())


class MODNetEvaluatorTests(unittest.TestCase):
    """The evaluator must run both sides itself, not compare hand-made mattes."""

    def _fixtures(self, temp):
        binding = importlib.import_module("samples.vision.modnet.runtime.python.model_binding")
        model = Path(temp) / "modnet.bin"
        model.write_bytes(b"fixture-modnet-model")
        selection = binding.resolve_selection(
            "x5", asset_id="x5:modnet:modnet_512x512_rgb.bin", model_path=str(model)
        )
        image_path = Path(__file__).resolve().parents[4] / "samples/vision/modnet/test_data/person.jpg"
        image = cv2.imread(str(image_path))
        assert image is not None
        return selection, image, image_path

    @staticmethod
    def _matte(offset: float = 0.0) -> np.ndarray:
        matte = np.linspace(0.0, 1.0, 512 * 512, dtype=np.float32).reshape(1, 1, 512, 512)
        if offset:
            matte = np.clip(matte + offset, 0.0, 1.0)
        return matte

    def _factory(self, mattes):
        state = {"index": 0}

        def make(path):
            value = mattes[min(state["index"], len(mattes) - 1)]
            state["index"] += 1
            return FakeRuntime(value)

        return make

    def _run(self, compare, selection, image, image_path, directory, factory):
        sdk = types.ModuleType("hbm_runtime")
        sdk.HB_HBMRuntime = FakeRuntime
        old_sdk = sys.modules.get("hbm_runtime")
        sys.modules["hbm_runtime"] = sdk
        try:
            with patch.object(compare, "require_execution_target", return_value="x5"):
                return compare.run_comparison(selection, image, image_path, directory,
                                              runtime_factory=factory)
        finally:
            if old_sdk is None:
                sys.modules.pop("hbm_runtime", None)
            else:
                sys.modules["hbm_runtime"] = old_sdk

    def test_evaluator_captures_both_sides_with_complete_identity(self):
        compare = importlib.import_module("samples.vision.modnet.evaluator.compare")
        with tempfile.TemporaryDirectory() as temp:
            selection, image, image_path = self._fixtures(temp)
            directory = Path(temp) / "success"
            summary = self._run(compare, selection, image, image_path, directory,
                                self._factory([self._matte()]))
            self.assertEqual(summary["return_code"], 0, summary.get("error"))
            self.assertTrue(summary["passed"])
            self.assertEqual(summary["source_ref"], "ac115717197920355fc390bb04299b20e6436864")
            self.assertTrue(summary["model_sha256"] and summary["image_sha256"])
            self.assertTrue(summary["code_sha256"])
            self.assertEqual(set(summary["metadata"]), {"legacy", "unified"})
            self.assertTrue(summary["started_utc"] and summary["finished_utc"])
            self.assertTrue(summary["argv"] and summary["cwd"])
            for filename, entry in summary["arrays"].items():
                self.assertTrue((directory / filename).is_file(), filename)
                self.assertEqual(len(entry["sha256"]), 64)
            self.assertTrue((directory / "comparison.json").is_file())
            self.assertTrue(all(summary["checks"].values()))

    def test_evaluator_reports_a_real_difference_instead_of_passing(self):
        compare = importlib.import_module("samples.vision.modnet.evaluator.compare")
        with tempfile.TemporaryDirectory() as temp:
            selection, image, image_path = self._fixtures(temp)
            directory = Path(temp) / "difference"
            summary = self._run(compare, selection, image, image_path, directory,
                                self._factory([self._matte(), self._matte(0.25)]))
            self.assertEqual(summary["return_code"], 1)
            self.assertFalse(summary["passed"])
            self.assertFalse(summary["checks"]["result.matte"])
            self.assertTrue((directory / "comparison.json").is_file())

    def test_evaluator_records_execution_failure_and_still_writes_evidence(self):
        compare = importlib.import_module("samples.vision.modnet.evaluator.compare")
        with tempfile.TemporaryDirectory() as temp:
            selection, image, image_path = self._fixtures(temp)
            directory = Path(temp) / "error"

            def explode(path):
                raise RuntimeError("fake SDK failure")

            with self.assertRaises(RuntimeError):
                self._run(compare, selection, image, image_path, directory, explode)
            payload = json.loads((directory / "comparison.json").read_text())
            self.assertEqual(payload["return_code"], 2)
            self.assertEqual(payload["error"]["type"], "RuntimeError")
            self.assertFalse(payload["passed"])


if __name__ == "__main__":
    unittest.main()
