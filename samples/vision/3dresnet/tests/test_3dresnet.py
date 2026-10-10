# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0

"""Host acceptance tests for the migrated 3DResNet video classifier."""
from __future__ import annotations

import contextlib
from dataclasses import replace
import importlib.util
import io
import json
import os
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from utils.py_utils.tests.legacy_platforms import legacy_path, legacy_tree  # noqa: E402

ROOT = Path(__file__).resolve().parents[4]
SAMPLE = ROOT / "samples/vision/3dresnet"
SOURCE = legacy_path("s/samples/vision/3dresnet/runtime/python/resnet3d.py")
RUNTIME = SAMPLE / "runtime/python"
MODEL = SAMPLE / "model"
PACKAGE = "samples.vision.3dresnet.runtime.python"
MODEL_PACKAGE = "samples.vision.3dresnet.model"
_LEGACY_SOURCE = None


def local_module(name):
    return importlib.import_module(f"{PACKAGE}.{name}")


class FakeRuntime:
    def __init__(self, input_name="physical_input", output_name="scores_out"):
        self.model_names = ["r3d_18"]
        self.input_names = {"r3d_18": [input_name]}
        self.input_shapes = {"r3d_18": {input_name: (1, 3, 16, 112, 112)}}
        self.input_dtypes = {"r3d_18": {input_name: "F32"}}
        self.output_names = {"r3d_18": [output_name]}
        self.output_shapes = {"r3d_18": {output_name: (1, 400)}}
        self.output_dtypes = {"r3d_18": {output_name: "F32"}}
        self.raw = np.linspace(-2.0, 2.0, 400, dtype=np.float32)[None, :]
        self.calls = []
        self.scheduling = None

    def run(self, inputs):
        self.calls.append(inputs)
        return {"r3d_18": {self.output_names["r3d_18"][0]: self.raw}}

    def set_scheduling_params(self, **kwargs):
        self.scheduling = kwargs


def metadata(runtime):
    input_name = runtime.input_names["r3d_18"][0]
    output_name = runtime.output_names["r3d_18"][0]
    return {
        "model_names": ["r3d_18"],
        "model_name": "r3d_18",
        "input_names": [input_name],
        "input_shapes": {input_name: (1, 3, 16, 112, 112)},
        "input_dtypes": {input_name: "F32"},
        "output_names": [output_name],
        "output_shapes": {output_name: (1, 400)},
        "output_dtypes": {output_name: "F32"},
    }


def load_legacy_source():
    global _LEGACY_SOURCE
    if _LEGACY_SOURCE is not None:
        return _LEGACY_SOURCE
    fake_hbm = types.ModuleType("hbm_runtime")
    fake_hbm.HB_HBMRuntime = object
    fake_hbm.QuantParams = object
    spec = importlib.util.spec_from_file_location("_resnet3d_legacy", SOURCE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    source_root = str(legacy_tree("s/utils").parent)
    sys.path.insert(0, source_root)
    try:
        with patch.dict(sys.modules, {"hbm_runtime": fake_hbm}):
            spec.loader.exec_module(module)
    finally:
        sys.path.remove(source_root)
    module_instance = module.ResNet3D.__new__(module.ResNet3D)
    module_instance.model_name = "r3d_18"
    module_instance.input_name = "physical_input"
    module_instance.output_name = "scores_out"
    module._test_instance = module_instance
    _LEGACY_SOURCE = module
    return module


class BindingTests(unittest.TestCase):
    def test_single_s100_asset_default_and_external_path_gate(self):
        resolve_selection = local_module("cli").resolve_selection

        selection = resolve_selection("s100")
        self.assertEqual(selection.asset.filename, "s100/r3d_18.hbm")
        self.assertEqual(selection.model_path, SAMPLE / "model/s100/r3d_18.hbm")
        with self.assertRaises(ValueError):
            resolve_selection("x5")
        with self.assertRaises(ValueError):
            resolve_selection("s100", model_path="/tmp/custom.hbm")
        self.assertEqual(
            resolve_selection(
                "s100",
                asset_id="s:3dresnet:s100/r3d_18.hbm",
                model_path="/tmp/custom.hbm",
            ).model_path,
            Path("/tmp/custom.hbm"),
        )

    def test_binding_uses_actual_dynamic_names_and_five_dim_input(self):
        bind_model = local_module("classification").bind_model
        resolve_selection = local_module("cli").resolve_selection

        runtime = FakeRuntime("not_input", "not_output")
        binding = bind_model(resolve_selection("s100"), metadata(runtime))
        self.assertEqual(binding.input_name, "not_input")
        self.assertEqual(binding.output_name, "not_output")
        self.assertEqual(binding.input_shape, (1, 3, 16, 112, 112))
        self.assertEqual(binding.output_shape, (1, 400))
        bad = metadata(runtime)
        bad["input_shapes"]["not_input"] = (1, 3, 224, 224)
        with self.assertRaises(ValueError):
            bind_model(resolve_selection("s100"), bad)

        selection = resolve_selection("s100")
        forged = replace(selection, asset=replace(selection.asset, url="https://example.invalid/forged.hbm"))
        with self.assertRaises(ValueError):
            bind_model(forged, metadata(runtime))


class TaskTests(unittest.TestCase):
    def make_task(self, runtime=None, **kwargs):
        classification = local_module("classification")
        resolve_selection = local_module("cli").resolve_selection

        runtime = runtime or FakeRuntime()
        selection = resolve_selection("s100")
        runner = classification.RuntimeModelRunner(selection, runtime=runtime)
        return classification.R3D18Classifier(selection, runner=runner, **kwargs), runtime

    def test_preprocess_matches_actual_source_and_casts_float32(self):
        legacy = load_legacy_source()._test_instance
        task, _ = self.make_task()
        clip = np.arange(1 * 3 * 16 * 112 * 112, dtype=np.float64).reshape(1, 3, 16, 112, 112) / 1000
        prepared = task.pre_process(clip)
        expected = legacy.pre_process(clip)["r3d_18"]["physical_input"]
        np.testing.assert_array_equal(prepared.tensors["physical_input"], expected)
        self.assertEqual(prepared.tensors["physical_input"].dtype, np.float32)
        self.assertEqual(prepared.tensors["physical_input"].shape, (1, 3, 16, 112, 112))

    def test_source_postprocess_matches_shared_topk(self):
        legacy = load_legacy_source()._test_instance
        task, runtime = self.make_task(top_k=5)
        source = legacy.post_process({"r3d_18": {"scores_out": runtime.raw}}, top_k=5)
        result = task.post_process({"scores_out": runtime.raw})
        self.assertEqual(result.as_legacy_tuple()[:2], (result.class_ids, result.scores))
        np.testing.assert_array_equal(result.class_ids, np.asarray([x[0] for x in source]))
        np.testing.assert_allclose(result.scores, np.asarray([x[1] for x in source], dtype=np.float32))

    def test_predict_explicit_stages_context_identity_and_raw_purity(self):
        task, runtime = self.make_task()
        first = np.zeros((1, 3, 16, 112, 112), dtype=np.float32)
        second = np.ones_like(first)
        a = task.pre_process(first)
        b = task.pre_process(second)
        c = task.pre_process(first)
        self.assertIsNot(a.context, b.context)
        self.assertIsNot(a.context, c.context)
        raw = task.forward(a.tensors)
        self.assertIs(raw["scores_out"], runtime.raw)
        explicit = task.post_process(raw)
        actual = task.predict(first)
        np.testing.assert_array_equal(actual.class_ids, explicit.class_ids)
        np.testing.assert_array_equal(actual.scores, explicit.scores)
        np.testing.assert_array_equal(runtime.raw, np.linspace(-2.0, 2.0, 400, dtype=np.float32)[None, :])

    def test_invalid_shapes_nonfinite_outputs_and_topk_metadata_reject(self):
        task, runtime = self.make_task()
        with self.assertRaises(ValueError):
            task.pre_process(np.zeros((1, 3, 15, 112, 112), dtype=np.float32))
        with self.assertRaises(ValueError):
            task.post_process({"scores_out": np.zeros((1, 399), dtype=np.float32)})
        bad = runtime.raw.copy()
        bad[0, 0] = np.nan
        with self.assertRaises(ValueError):
            task.post_process({"scores_out": bad})
        with self.assertRaises(ValueError):
            self.make_task(top_k=0)
        with self.assertRaises(ValueError):
            self.make_task(top_k=401)


class ReadableInterfaceTests(unittest.TestCase):
    """The canonical preprocess/infer/postprocess names drive predict."""

    def make_task(self, **kwargs):
        return TaskTests().make_task(**kwargs)

    def test_canonical_stage_names_exist_and_legacy_names_delegate(self):
        task, _ = self.make_task()
        for name in ("preprocess", "infer", "postprocess", "predict"):
            self.assertTrue(callable(getattr(task, name, None)), name)
        clip = np.zeros((1, 3, 16, 112, 112), dtype=np.float32)
        prepared_new, prepared_old = task.preprocess(clip), task.pre_process(clip)
        np.testing.assert_array_equal(
            prepared_new.tensors["physical_input"], prepared_old.tensors["physical_input"]
        )
        self.assertIsNotNone(prepared_new.context)
        np.testing.assert_array_equal(
            task.infer(prepared_new)["scores_out"], task.forward(prepared_old)["scores_out"]
        )
        explicit_new = task.postprocess({"scores_out": np.zeros((1, 400), np.float32)})
        explicit_old = task.post_process({"scores_out": np.zeros((1, 400), np.float32)})
        np.testing.assert_array_equal(explicit_new.class_ids, explicit_old.class_ids)
        np.testing.assert_array_equal(explicit_new.scores, explicit_old.scores)

    def test_predict_routes_through_canonical_stages_with_single_runner_call(self):
        task, runtime = self.make_task()
        routed = []
        for canonical in ("preprocess", "infer", "postprocess"):
            original = getattr(task, canonical)

            def spy(*args, _original=original, _canonical=canonical, **kwargs):
                routed.append(_canonical)
                return _original(*args, **kwargs)

            setattr(task, canonical, spy)
        clip = np.zeros((1, 3, 16, 112, 112), dtype=np.float32)
        calls_before = len(runtime.calls)
        result = task.predict(clip)
        self.assertEqual(routed, ["preprocess", "infer", "postprocess"])
        self.assertEqual(len(runtime.calls), calls_before + 1)
        self.assertEqual(len(result.class_ids), 5)

    def test_legacy_stage_names_delegate_to_canonical_implementations(self):
        task, runtime = self.make_task()
        routed = []
        for canonical in ("preprocess", "infer", "postprocess"):
            original = getattr(task, canonical)

            def spy(*args, _original=original, _canonical=canonical, **kwargs):
                routed.append(_canonical)
                return _original(*args, **kwargs)

            setattr(task, canonical, spy)
        clip = np.zeros((1, 3, 16, 112, 112), dtype=np.float32)
        prepared = task.pre_process(clip)
        raw = task.forward(prepared)
        task.post_process(raw)
        self.assertEqual(routed, ["preprocess", "infer", "postprocess"])
        self.assertEqual(len(runtime.calls), 1)

    def test_predict_equals_canonical_manual_composition_and_per_call_state(self):
        task, runtime = self.make_task()
        first = np.zeros((1, 3, 16, 112, 112), dtype=np.float32)
        second = np.full((1, 3, 16, 112, 112), 0.5, dtype=np.float32)
        manual = task.postprocess(task.infer(task.preprocess(first)))
        self.assertEqual(manual.class_ids.tolist(), task.predict(first).class_ids.tolist())
        np.testing.assert_allclose(manual.scores, task.predict(first).scores)
        runtime.raw[0, 7] = 40.0
        changed = task.predict(second)
        self.assertEqual(int(changed.class_ids[0]), 7)
        unchanged = task.predict(first)
        self.assertEqual(int(unchanged.class_ids[0]), 7)
        self.assertEqual(len(runtime.calls), 5)


class LabelsAndCLITests(unittest.TestCase):
    def test_clean_repo_root_importlib_loads_task_and_runner(self):
        code = (
            "import importlib; "
            "importlib.import_module('samples.vision.3dresnet.runtime.python.classification')"
        )
        environment = os.environ.copy()
        environment.pop("PYTHONPATH", None)
        proc = subprocess.run(
            [sys.executable, "-c", code],
            cwd=ROOT,
            env=environment,
            capture_output=True,
            text=True,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)

    def test_labels_keep_source_name_to_id_decode(self):
        load_labels = local_module("cli").load_labels

        labels = load_labels(SAMPLE / "test_data/kinetics_classnames.json")
        self.assertEqual(len(labels), 400)
        self.assertEqual(labels[5], "archery")
        self.assertEqual(labels[290], "sharpening knives")

    def test_cli_help_list_and_dryrun_are_sdk_free(self):
        for args in (("--help",), ("--list-models",), ("--dry-run", "--target", "s100")):
            proc = subprocess.run(
                [sys.executable, str(SAMPLE / "runtime/python/main.py"), *args],
                cwd="/tmp", capture_output=True, text=True,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
        proc = subprocess.run(
            [sys.executable, str(SAMPLE / "runtime/python/main.py"), "--dry-run"],
            cwd="/tmp", capture_output=True, text=True,
        )
        self.assertEqual(proc.returncode, 2)

    def test_cli_fixture_pipeline_reads_labels_and_emits_json(self):
        main = local_module("main")
        classification = local_module("classification")

        runtime = FakeRuntime()
        real_runner = classification.RuntimeModelRunner
        with tempfile.TemporaryDirectory() as temp, patch("utils.py_utils.platforms.detect_target", return_value="s100"), patch.object(
            classification, "RuntimeModelRunner", lambda selection: real_runner(selection, runtime=runtime)
        ):
            model_path = Path(temp) / "r3d_18.hbm"
            model_path.write_bytes(b"fixture")
            clip_path = Path(temp) / "clip.npy"
            np.save(clip_path, np.zeros((1, 3, 16, 112, 112), dtype=np.float32))
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                code = main.main([
                    "--target", "s100",
                    "--asset-id", "s:3dresnet:s100/r3d_18.hbm",
                    "--model-path", str(model_path),
                    "--test-clip", str(clip_path),
                    "--label-file", str(SAMPLE / "test_data/kinetics_classnames.json"),
                ])
            self.assertEqual(code, 0)
            report = json.loads(output.getvalue())
            self.assertEqual(len(report["predictions"]), 5)
            self.assertIn("label", report["predictions"][0])

    def test_cli_rejects_unknown_identity_before_runner(self):
        main = local_module("main")
        classification = local_module("classification")

        with patch("utils.py_utils.platforms.detect_target", return_value=None), patch.object(
            classification, "RuntimeModelRunner", side_effect=AssertionError("runner must not construct")
        ):
            self.assertEqual(main.main([
                "--target", "s100",
                "--asset-id", "s:3dresnet:s100/r3d_18.hbm",
                "--model-path", "/tmp/r3d_18.hbm",
            ]), 2)

    def test_download_delegates_exact_manifest_asset(self):
        download = importlib.import_module(f"{MODEL_PACKAGE}.download")

        calls = []
        with tempfile.TemporaryDirectory() as temp, patch.object(
            download, "download_asset", side_effect=lambda asset, path: calls.append((asset.reference, Path(path))) or "digest"
        ):
            download.download_target("s100", temp)
        self.assertEqual(calls[0][0], "s:3dresnet:s100/r3d_18.hbm")
        self.assertEqual(calls[0][1].name, "r3d_18.hbm")


if __name__ == "__main__":
    unittest.main()


class SimplifiedRuntimeTests(unittest.TestCase):
    """2026-10-08 runtime simplification boundary.

    Selection/catalog duties live in ``cli.py``; ``classification.py`` owns the
    named model class plus the sample runner and clip preparation; the
    per-sample ``model_binding``/``model_runner``/``tensor_io`` forwarding
    modules are gone. Kinetics-400 label decoding lives in ``cli.py``.
    """

    def test_classifier_constructs_from_selection_and_runs_stages(self):
        resolve_selection = local_module("cli").resolve_selection
        classification = local_module("classification")
        runtime = FakeRuntime()
        selection = resolve_selection("s100")
        model = classification.R3D18Classifier(
            selection, top_k=3,
            runner=classification.RuntimeModelRunner(selection, runtime=runtime))
        result = model.predict(np.zeros((1, 3, 16, 112, 112), np.float32))
        self.assertEqual(len(result.class_ids), 3)
        prepared = model.preprocess(np.zeros((1, 3, 16, 112, 112), np.float32))
        manual = model.postprocess(model.infer(prepared))
        np.testing.assert_array_equal(result.class_ids, manual.class_ids)
        np.testing.assert_allclose(result.scores, manual.scores)

    def test_split_forwarding_modules_are_removed(self):
        base = SAMPLE / "runtime/python"
        for name in ("model_binding.py", "model_runner.py", "tensor_io.py"):
            self.assertFalse((base / name).exists(), name)
