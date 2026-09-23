# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[4]


class FakeRuntime:
    model_names = ["lpr"]
    input_names = {"lpr": ["input"]}
    input_shapes = {"lpr": {"input": (1, 3, 24, 94)}}
    input_dtypes = {"lpr": {"input": "float32"}}
    output_names = {"lpr": ["output"]}
    output_shapes = {"lpr": {"output": (1, 68, 18)}}
    input_strides = {"lpr": {}}
    output_strides = {"lpr": {}}
    output_dtypes = {"lpr": {"output": "float32"}}
    output_quants = {"lpr": {}}

    def __init__(self, output: np.ndarray):
        self.output = output
        self.calls = []

    def run(self, tensors):
        self.calls.append(tensors)
        return {"lpr": {"output": self.output}}

    def set_scheduling_params(self, **kwargs):
        self.scheduling = kwargs


class LPRNetTests(unittest.TestCase):
    def test_ctc_decoder_matches_fixed_source_for_numeric_fixture(self):
        source_path = ROOT / "platforms/x5/samples/vision/lprnet/runtime/python/lprnet.py"
        fake_hbm = types.ModuleType("hbm_runtime")
        old_hbm = sys.modules.get("hbm_runtime")
        sys.modules["hbm_runtime"] = fake_hbm
        try:
            spec = importlib.util.spec_from_file_location("legacy_lprnet", source_path)
            module = importlib.util.module_from_spec(spec)
            assert spec.loader is not None
            sys.modules["legacy_lprnet"] = module
            spec.loader.exec_module(module)
            rng = np.random.default_rng(7)
            logits = rng.normal(size=(68, 18)).astype(np.float32)
            unified = importlib.import_module("samples.vision.lprnet.runtime.python.lprnet")
            self.assertEqual(module.decode_plate(logits), unified.decode_plate(logits))
        finally:
            if old_hbm is None:
                sys.modules.pop("hbm_runtime", None)
            else:
                sys.modules["hbm_runtime"] = old_hbm
            sys.modules.pop("legacy_lprnet", None)

    def test_ctc_decoder_removes_repeats_and_blank(self):
        decode = importlib.import_module(
            "samples.vision.lprnet.runtime.python.lprnet"
        ).decode_plate
        logits = np.full((68, 18), -10.0, dtype=np.float32)
        # 京, 京, blank, A, A, 7 -> 京A7
        for time, index in enumerate((0, 0, 67, 41, 41, 38)):
            logits[index, time] = 10.0
        logits[67, 6:] = 10.0
        self.assertEqual(decode(logits), "京A7")

    def test_binding_accepts_source_metadata_and_rejects_shape_or_dtype(self):
        binding = importlib.import_module(
            "samples.vision.lprnet.runtime.python.model_binding"
        )
        selection = binding.resolve_selection("x5")
        good = {
            "model_names": ("lpr",),
            "model_name": "lpr",
            "input_names": ("input",),
            "input_shapes": {"input": (1, 3, 24, 94)},
            "input_dtypes": {"input": "float32"},
            "output_names": ("output",),
            "output_shapes": {"output": (1, 68, 18)},
            "output_dtypes": {"output": "float32"},
        }
        self.assertEqual(binding.bind_model(selection, good).input_name, "input")
        for field, value in (
            ("input_shapes", {"input": (1, 3, 24, 95)}),
            ("output_dtypes", {"output": "int8"}),
        ):
            bad = dict(good)
            bad[field] = value
            with self.assertRaises(binding.MetadataMismatchError):
                binding.bind_model(selection, bad)

    def test_task_preserves_raw_logits_and_predict_matches_explicit_stages(self):
        runtime_mod = importlib.import_module(
            "samples.vision.lprnet.runtime.python.model_runner"
        )
        task_mod = importlib.import_module(
            "samples.vision.lprnet.runtime.python.lprnet"
        )
        binding_mod = importlib.import_module(
            "samples.vision.lprnet.runtime.python.model_binding"
        )
        selection = binding_mod.resolve_selection("x5")
        output = np.full((1, 68, 18), -4.0, dtype=np.float32)
        output[:, 0, 0] = 4.0
        fake = FakeRuntime(output)
        runner = runtime_mod.RuntimeModelRunner(
            selection, runtime=fake
        )
        runner.load()
        task = task_mod.LPRNetTask(runner, runner.binding)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "input.dat"
            source = np.arange(1 * 3 * 24 * 94, dtype=np.float32).reshape(1, 3, 24, 94)
            source.tofile(path)
            prepared = task.pre_process(path)
            raw = task.forward(prepared.tensors)
            self.assertTrue(np.array_equal(raw, output))
            explicit = task.post_process(raw)
            self.assertEqual(explicit, task.predict(path))
            self.assertEqual(fake.calls[0]["lpr"]["input"].shape, source.shape)

    def test_external_model_path_requires_exact_asset_id(self):
        binding = importlib.import_module(
            "samples.vision.lprnet.runtime.python.model_binding"
        )
        with self.assertRaises(binding.BindingError):
            binding.resolve_selection("x5", model_path="/tmp/lpr.bin")

    def test_cli_list_and_dry_run_do_not_construct_runtime(self):
        main = importlib.import_module("samples.vision.lprnet.runtime.python.main")
        with patch.object(main, "RuntimeModelRunner", side_effect=AssertionError):
            self.assertEqual(main.main(["--list-models"]), 0)
            self.assertEqual(main.main(["--dry-run", "--target", "x5"]), 0)

    def test_download_uses_explicit_asset_helper_without_network_in_test(self):
        module = importlib.import_module("samples.vision.lprnet.model.download")
        with tempfile.TemporaryDirectory() as temp:
            with patch.object(module, "download_asset", return_value="fixture-digest") as download:
                self.assertEqual(module.main(["--target", "x5", "--output-dir", temp]), 0)
                download.assert_called_once()
                self.assertEqual(download.call_args.args[0].reference, "x5:lprnet:lpr.bin")


if __name__ == "__main__":
    unittest.main()
