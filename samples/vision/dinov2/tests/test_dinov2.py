# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0

"""Host acceptance tests for the DINOv2 migration.

The runtime in these tests is an injected fixture.  The legacy DINOv2 module
is imported from the fixed S source for numerical and preprocessing parity;
only its SDK module is stubbed.
"""
from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
SAMPLE = ROOT / "samples/vision/dinov2"
SOURCE = ROOT / "platforms/s/samples/vision/dinov2/runtime/python/dinov2.py"
_LEGACY_SOURCE = None


class _EnumLike:
    def __init__(self, name: str):
        self.name = name


class QuantInfo:
    def __init__(self, scale, zero_point, axis=0, quant_type="SCALE"):
        self.scale = np.asarray(scale, dtype=np.float32)
        self.zero_point = np.asarray(zero_point, dtype=np.float32)
        self.axis = axis
        self.quant_type = _EnumLike(quant_type)


def metadata(dtype="I16"):
    output_dtype = {"I16": "I16", "F32": "F32"}[dtype]
    quants = {
        "cls_feat": QuantInfo(0.25, 3),
        "patch_feat": QuantInfo([0.5] * 384, [1] * 384, axis=2),
    }
    return {
        "model_names": ["dinov2"],
        "model_name": "dinov2",
        "input_names": ["input"],
        "input_shapes": {"input": (1, 3, 224, 224)},
        "input_dtypes": {"input": "F32"},
        "output_names": ["cls_feat", "patch_feat"],
        "output_shapes": {
            "cls_feat": (1, 384),
            "patch_feat": (1, 256, 384),
        },
        "output_dtypes": {
            "cls_feat": output_dtype,
            "patch_feat": output_dtype,
        },
        "output_quants": quants,
    }


class FakeRuntime:
    def __init__(self, dtype="I16"):
        facts = metadata(dtype)
        for key, value in facts.items():
            setattr(self, key, value)
        self.calls = []
        self.scheduling = {}
        np_dtype = np.int16 if dtype == "I16" else np.float32
        self.raw = {
            "cls_feat": np.ones((1, 384), dtype=np_dtype),
            "patch_feat": np.ones((1, 256, 384), dtype=np_dtype),
        }

    def run(self, inputs):
        self.calls.append(inputs)
        return {"dinov2": self.raw}

    def set_scheduling_params(self, **kwargs):
        self.scheduling = kwargs


def load_legacy_source():
    """Import the actual fixed source wrapper with only SDK imports stubbed."""
    global _LEGACY_SOURCE
    if _LEGACY_SOURCE is not None:
        return _LEGACY_SOURCE
    fake_hbm = types.ModuleType("hbm_runtime")
    fake_hbm.HB_HBMRuntime = object
    fake_hbm.QuantParams = object
    spec = importlib.util.spec_from_file_location("_dinov2_legacy", SOURCE)
    module = importlib.util.module_from_spec(spec)
    source_utils = str(ROOT / "platforms/s")
    previous_module = sys.modules.get(spec.name)
    sys.modules[spec.name] = module
    sys.path.insert(0, source_utils)
    try:
        with patch.dict(sys.modules, {"hbm_runtime": fake_hbm}):
            spec.loader.exec_module(module)
    finally:
        sys.path.remove(source_utils)
        if previous_module is None:
            sys.modules.pop(spec.name, None)
        else:
            sys.modules[spec.name] = previous_module
    _LEGACY_SOURCE = module
    return _LEGACY_SOURCE


def legacy_wrapper(module, output="cls_feat"):
    wrapper = module.Dinov2.__new__(module.Dinov2)
    wrapper.model_name = "dinov2"
    wrapper.cfg = module.Dinov2Config("unused", output=output)
    wrapper.model = types.SimpleNamespace(output_quants={"dinov2": {
        "cls_feat": QuantInfo(0.25, 3),
        "patch_feat": QuantInfo([0.5] * 384, [1] * 384, axis=2),
    }})
    return wrapper


class BindingTests(unittest.TestCase):
    def test_three_exact_target_assets_and_no_unknown_fallback(self):
        from samples.vision.dinov2.runtime.python.model_binding import (
            list_available_assets,
            resolve_selection,
        )

        expected = {
            "s100": "nash-e/dinov2_vits14_224_int16_nashe.hbm",
            "s100p": "nash-m/dinov2_vits14_224_int16_nashm.hbm",
            "s600": "nash-p/dinov2_vits14_224_int16_nashp.hbm",
        }
        self.assertEqual(
            {asset.filename for asset in list_available_assets()}, set(expected.values())
        )
        for target, filename in expected.items():
            selection = resolve_selection(target)
            self.assertEqual(selection.asset.filename, filename)
            self.assertEqual(selection.target, target)
        with self.assertRaises(ValueError):
            resolve_selection("x5")
        with self.assertRaises(ValueError):
            resolve_selection("auto")

    def test_external_model_path_requires_exact_asset_id(self):
        from samples.vision.dinov2.runtime.python.model_binding import resolve_selection

        with self.assertRaises(ValueError):
            resolve_selection("s100", model_path="/tmp/custom.hbm")
        selection = resolve_selection(
            "s100",
            asset_id="s:dinov2:nash-e/dinov2_vits14_224_int16_nashe.hbm",
            model_path="/tmp/custom.hbm",
        )
        self.assertEqual(selection.model_path, Path("/tmp/custom.hbm"))

    def test_binding_requires_two_outputs_and_declares_per_output_transform(self):
        from samples.vision.dinov2.runtime.python.model_binding import (
            bind_model,
            resolve_selection,
        )

        selection = resolve_selection("s100")
        binding = bind_model(selection, metadata("I16"))
        self.assertEqual(binding.output_shapes["cls_feat"], (1, 384))
        self.assertEqual(binding.output_shapes["patch_feat"], (1, 256, 384))
        self.assertEqual(binding.output_transforms["cls_feat"], "dequant")
        bad = metadata("I16")
        bad["output_names"] = ["cls_feat"]
        with self.assertRaises(ValueError):
            bind_model(selection, bad)


class TaskTests(unittest.TestCase):
    def task(self, dtype="I16", target="s100", output="cls_feat"):
        from samples.vision.dinov2.runtime.python.embedding import DINOv2Task
        from samples.vision.dinov2.runtime.python.model_binding import (
            bind_model,
            resolve_selection,
        )
        from samples.vision.dinov2.runtime.python.model_runner import RuntimeModelRunner

        runtime = FakeRuntime(dtype)
        selection = resolve_selection(target)
        binding = bind_model(selection, metadata(dtype))
        runner = RuntimeModelRunner(selection, runtime=runtime)
        runner.binding = binding
        return DINOv2Task(runner, binding, output), runtime

    def test_preprocess_matches_legacy_source_and_context_is_per_call(self):
        legacy = load_legacy_source()
        old = legacy_wrapper(legacy)
        task, _ = self.task()
        rng = np.random.default_rng(3)
        prepared = []
        for shape in ((31, 59, 3), (92, 33, 3), (31, 59, 3)):
            image = rng.integers(0, 256, shape, dtype=np.uint8)
            got = task.pre_process(image)
            expected = old.pre_process(image)["dinov2"]["input"]
            np.testing.assert_array_equal(got.tensors["input"], expected)
            prepared.append(got)
        self.assertEqual(prepared[0].context, prepared[2].context)
        self.assertNotEqual(prepared[0].context, prepared[1].context)

    def test_forward_is_raw_and_predict_equals_explicit_three_steps(self):
        task, runtime = self.task()
        image = np.zeros((32, 81, 3), dtype=np.uint8)
        prepared = task.pre_process(image)
        raw = task.forward(prepared.tensors)
        self.assertIs(raw["cls_feat"], runtime.raw["cls_feat"])
        explicit = task.post_process(task.forward(prepared.tensors))
        actual = task.predict(image)
        np.testing.assert_array_equal(actual, explicit)
        self.assertEqual(actual.dtype, np.float32)
        self.assertFalse(np.shares_memory(actual, runtime.raw["cls_feat"]))

    def test_int16_per_tensor_and_per_channel_match_real_source_helper(self):
        legacy = load_legacy_source()
        old = legacy_wrapper(legacy, output="cls_feat")
        task, runtime = self.task(output="cls_feat")
        runtime.raw["cls_feat"][:,:4] = np.array([[-10, 0, 10, 20]], dtype=np.int16)
        expected_cls = old.post_process(
            {"dinov2": {"cls_feat": runtime.raw["cls_feat"], "patch_feat": runtime.raw["patch_feat"]}}
        )
        actual_cls = task.post_process(task.forward(task.pre_process(np.zeros((224, 224, 3), np.uint8)).tensors))
        np.testing.assert_array_equal(actual_cls, expected_cls)

        from samples._shared.quantization import dequantize_tensor

        q = np.array([[1, 2], [10, 20]], dtype=np.int16)
        info = QuantInfo([0.5, 1.5], [3, 5], axis=1)
        np.testing.assert_array_equal(
            dequantize_tensor(q, info), legacy.postprocess.dequantize_tensor(q, info)
        )

        old_patch = legacy_wrapper(legacy, output="patch_feat")
        patch_task, patch_runtime = self.task(output="patch_feat")
        patch_runtime.raw["patch_feat"][0, 0, :4] = np.array([0, 2, 4, 6], dtype=np.int16)
        expected_patch = old_patch.post_process({
            "dinov2": {
                "cls_feat": patch_runtime.raw["cls_feat"],
                "patch_feat": patch_runtime.raw["patch_feat"],
            }
        })
        actual_patch = patch_task.post_process(patch_task.forward(
            patch_task.pre_process(np.zeros((224, 224, 3), np.uint8)).tensors
        ))
        np.testing.assert_array_equal(actual_patch, expected_patch)

    def test_all_targets_and_outputs_bind_both_raw_dtypes(self):
        from samples.vision.dinov2.runtime.python.model_binding import bind_model, resolve_selection
        from samples.vision.dinov2.runtime.python.embedding import DINOv2Task
        from samples.vision.dinov2.runtime.python.model_runner import RuntimeModelRunner

        for target in ("s100", "s100p", "s600"):
            for dtype in ("I16", "F32"):
                for output, shape in (("cls_feat", (1, 384)), ("patch_feat", (1, 256, 384))):
                    runtime = FakeRuntime(dtype)
                    selection = resolve_selection(target)
                    binding = bind_model(selection, metadata(dtype))
                    runner = RuntimeModelRunner(selection, runtime=runtime)
                    runner.binding = binding
                    result = DINOv2Task(runner, binding, output).predict(
                        np.zeros((224, 224, 3), dtype=np.uint8)
                    )
                    self.assertEqual(result.shape, shape)
                    self.assertEqual(result.dtype, np.float32)

    def test_non_scale_integer_is_cast_to_final_float32_without_activation(self):
        legacy = load_legacy_source()
        task, runtime = self.task()
        info = QuantInfo(0.5, 7, quant_type="NON_SCALE")
        task.binding.output_quants["cls_feat"] = info
        runtime.output_quants["cls_feat"] = info
        out = task.post_process({"cls_feat": runtime.raw["cls_feat"], "patch_feat": runtime.raw["patch_feat"]})
        self.assertEqual(out.dtype, np.float32)
        expected = legacy.postprocess.dequantize_tensor(runtime.raw["cls_feat"], info).astype(np.float32)
        np.testing.assert_array_equal(out, expected)

    def test_bad_inputs_outputs_nonfinite_and_shape_are_rejected(self):
        task, _ = self.task()
        for image in (
            np.zeros((0, 10, 3), np.uint8),
            np.zeros((10, 10), np.uint8),
            np.zeros((10, 10, 3), np.float32),
        ):
            with self.assertRaises(ValueError):
                task.pre_process(image)
        with self.assertRaises(ValueError):
            task.post_process({"cls_feat": np.zeros((1, 2), np.int16), "patch_feat": np.zeros((1, 256, 384), np.int16)})


class RunnerAndCLITests(unittest.TestCase):
    def test_runner_metadata_and_scheduling_are_real_fixture_pipeline(self):
        from samples.vision.dinov2.runtime.python.model_binding import resolve_selection
        from samples.vision.dinov2.runtime.python.model_runner import RuntimeModelRunner

        runtime = FakeRuntime("I16")
        runner = RuntimeModelRunner(resolve_selection("s100"), runtime=runtime)
        runner.load()
        runner.set_scheduling_params(priority=2, bpu_cores=[0])
        self.assertEqual(runtime.scheduling["priority"], {"dinov2": 2})
        output = runner({"input": np.zeros((1, 3, 224, 224), np.float32)})
        self.assertIs(output["cls_feat"], runtime.raw["cls_feat"])

    def test_cli_help_list_and_dry_run_are_sdk_free(self):
        for args in (("--help",), ("--list-models",), ("--dry-run", "--target", "s600")):
            proc = subprocess.run(
                [sys.executable, str(SAMPLE / "runtime/python/main.py"), *args],
                cwd="/tmp",
                capture_output=True,
                text=True,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
        proc = subprocess.run(
            [sys.executable, str(SAMPLE / "runtime/python/main.py"), "--dry-run"],
            cwd="/tmp",
            capture_output=True,
            text=True,
        )
        self.assertEqual(proc.returncode, 2)

    def test_cli_runs_fixture_pipeline_writes_exact_output_and_cosine(self):
        from samples.vision.dinov2.runtime.python import main, model_runner

        runtime = FakeRuntime("F32")
        real_runner = model_runner.RuntimeModelRunner
        with patch("samples._shared.platforms.detect_target", return_value="s100"), patch.object(
            model_runner, "RuntimeModelRunner", lambda selection: RuntimeModelRunnerFixture(selection, runtime, real_runner)
        ):
            with tempfile.TemporaryDirectory() as temp:
                temp_path = Path(temp)
                model_path = temp_path / "model.hbm"
                model_path.write_bytes(b"fixture")
                first = temp_path / "first.jpg"
                second = temp_path / "second.jpg"
                first.write_bytes((SAMPLE / "test_data/dog.jpg").read_bytes())
                second.write_bytes((SAMPLE / "test_data/bus.jpg").read_bytes())
                output = temp_path / "nested/result.npy.bin"
                text = io.StringIO()
                with contextlib.redirect_stdout(text):
                    code = main.main([
                        "--target", "s100",
                        "--asset-id", "s:dinov2:nash-e/dinov2_vits14_224_int16_nashe.hbm",
                        "--model-path", str(model_path),
                        "--test-img", str(first),
                        "--second-img", str(second),
                        "--output-file", str(output),
                    ])
                self.assertEqual(code, 0)
                self.assertTrue(output.is_file())
                self.assertEqual(np.load(output, allow_pickle=False).shape, (1, 384))
                summary = json.loads(text.getvalue().split("Feature tensor saved:")[0])
                self.assertEqual(summary["output"], "cls_feat")
                self.assertIn("cosine_similarity", summary)

    def test_cli_rejects_unknown_identity_before_runner(self):
        from samples.vision.dinov2.runtime.python import main, model_runner

        with patch("samples._shared.platforms.detect_target", return_value=None), patch.object(
            model_runner, "RuntimeModelRunner", side_effect=AssertionError("runner must not be constructed")
        ):
            code = main.main([
                "--target", "s100",
                "--asset-id", "s:dinov2:nash-e/dinov2_vits14_224_int16_nashe.hbm",
                "--model-path", "/tmp/does-not-matter.hbm",
            ])
        self.assertEqual(code, 2)

    def test_download_delegates_exact_asset_for_all_targets(self):
        from samples.vision.dinov2.model import download

        calls = []

        def fake_download(asset, destination):
            calls.append((asset.reference, Path(destination)))
            return "observed-digest"

        with tempfile.TemporaryDirectory() as temp, patch.object(download, "download_asset", side_effect=fake_download):
            for target, suffix in (("s100", "nashe"), ("s100p", "nashm"), ("s600", "nashp")):
                download.download_target(target, temp)
        self.assertEqual([ref for ref, _ in calls], [
            "s:dinov2:nash-e/dinov2_vits14_224_int16_nashe.hbm",
            "s:dinov2:nash-m/dinov2_vits14_224_int16_nashm.hbm",
            "s:dinov2:nash-p/dinov2_vits14_224_int16_nashp.hbm",
        ])
        self.assertEqual([path.name for _, path in calls], [
            "dinov2_vits14_224_int16_nashe.hbm",
            "dinov2_vits14_224_int16_nashm.hbm",
            "dinov2_vits14_224_int16_nashp.hbm",
        ])


class RuntimeModelRunnerFixture:
    """Adapter used only to exercise main's real image/summary pipeline."""

    def __init__(self, selection, runtime, runner_class):
        self._runner = runner_class(selection, runtime=runtime)

    def __getattr__(self, name):
        return getattr(self._runner, name)

    def __call__(self, inputs):
        return self._runner(inputs)


if __name__ == "__main__":
    unittest.main()
