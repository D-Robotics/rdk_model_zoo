# Copyright (c) 2026 D-Robotics Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Host contract tests for the migrated FCOS sample.

Fixtures use real source post-processing for numerical comparison.  The only
source import seam is a minimal ``hbm_runtime.QuantParams`` type stub because
the source helper imports that annotation at module import time; no source
numeric helper is replaced.
"""

from __future__ import annotations

import importlib.util
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
SAMPLE = ROOT / "samples" / "vision" / "fcos"
SOURCE = ROOT / "platforms" / "x5" / "samples" / "vision" / "fcos" / "runtime" / "python" / "fcos_det.py"


class QuantType:
    def __init__(self, name: str = "SCALE"):
        self.name = name


class Quant:
    def __init__(self, scale=1.0, zero_point=0.0, axis=0, quant_type="SCALE"):
        self.quant_type = QuantType(quant_type)
        self.scale = np.asarray(scale, dtype=np.float32)
        self.zero_point = np.asarray(zero_point, dtype=np.float32)
        self.axis = axis


def _source_class():
    module_name = "fcos_source_contract_fixture"
    if module_name in sys.modules:
        return sys.modules[module_name].FCOSDetect
    sdk = types.ModuleType("hbm_runtime")
    sdk.QuantParams = Quant
    with patch.dict(sys.modules, {"hbm_runtime": sdk}):
        spec = importlib.util.spec_from_file_location(module_name, SOURCE)
        module = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    return module.FCOSDetect


def _shapes(size: int = 512):
    return {
        **{
            f"cls_{stride}": (1, size // stride, size // stride, 80)
            for stride in (8, 16, 32, 64, 128)
        },
        **{
            f"box_{stride}": (1, size // stride, size // stride, 4)
            for stride in (8, 16, 32, 64, 128)
        },
        **{
            f"center_{stride}": (1, size // stride, size // stride, 1)
            for stride in (8, 16, 32, 64, 128)
        },
    }


def metadata_for(variant="efficientnetb0", *, bad=False, dtype="int8", per_channel=False):
    size = {"efficientnetb0": 512, "efficientnetb2": 768, "efficientnetb3": 896}[variant]
    shapes = _shapes(size)
    if bad:
        shapes.pop("center_128")
    names = tuple(shapes)
    quants = {name: Quant(0.1, 0) for name in names}
    if per_channel:
        quants["cls_8"] = Quant(np.linspace(0.05, 0.2, 80, dtype=np.float32), np.zeros(80, dtype=np.float32), axis=3)
    return {
        "model_name": f"fcos_{variant}",
        "input_names": ["input"],
        "input_shapes": {"input": (1, 3, size, size)},
        "input_dtypes": {"input": "NV12"},
        "output_names": names,
        "output_shapes": shapes,
        "output_dtypes": {name: dtype for name in names},
        "output_quants": quants,
    }


def fixture_outputs(size=512):
    shapes = _shapes(size)
    outputs = {}
    for name, shape in shapes.items():
        outputs[name] = np.full(shape, -12, dtype=np.int8)
    # One valid candidate at each level; values are logits before source
    # sigmoid and distances before source stride multiplication.
    for index, stride in enumerate((8, 16, 32, 64, 128)):
        outputs[f"cls_{stride}"][0, 0, 0, index] = 10
        outputs[f"center_{stride}"][0, 0, 0, 0] = 10
        outputs[f"box_{stride}"][0, 0, 0, :] = 1
    return outputs


class FcosContractTests(unittest.TestCase):
    def test_three_manifest_variants_have_exact_asset_identity(self):
        from samples.vision.fcos.runtime.python.model_binding import list_available_assets

        assets = list_available_assets("x5")
        self.assertEqual([asset.variant for asset in assets], ["efficientnetb0", "efficientnetb2", "efficientnetb3"])
        self.assertEqual(
            [asset.asset_id for asset in assets],
            [
                "x5:fcos:fcos_efficientnetb0_detect_512x512_bayese_nv12.bin",
                "x5:fcos:fcos_efficientnetb2_detect_768x768_bayese_nv12.bin",
                "x5:fcos:fcos_efficientnetb3_detect_896x896_bayese_nv12.bin",
            ],
        )

    def test_each_variant_binds_five_by_three_output_families(self):
        from samples.vision.fcos.runtime.python.model_binding import bind_model, resolve_selection

        for variant in ("efficientnetb0", "efficientnetb2", "efficientnetb3"):
            with self.subTest(variant=variant):
                selection = resolve_selection("x5", variant=variant)
                binding = bind_model(selection, metadata_for(variant))
                self.assertEqual(len(binding.cls_output_names), 5)
                self.assertEqual(len(binding.box_output_names), 5)
                self.assertEqual(len(binding.center_output_names), 5)
                self.assertEqual(binding.contract.output_transform, "dequant")

    def test_binding_rejects_missing_or_wrong_variant_metadata(self):
        from samples.vision.fcos.runtime.python.model_binding import bind_model, resolve_selection

        with self.assertRaises(ValueError):
            bind_model(resolve_selection("x5", variant="efficientnetb0"), metadata_for("efficientnetb0", bad=True))
        with self.assertRaises(ValueError):
            bind_model(resolve_selection("x5", variant="efficientnetb2"), metadata_for("efficientnetb0"))

    def test_external_model_path_requires_exact_asset_reference(self):
        from samples.vision.fcos.runtime.python.model_binding import resolve_selection

        with self.assertRaises(ValueError):
            resolve_selection("x5", model_path="/tmp/fcos.bin")
        selection = resolve_selection(
            "x5",
            asset_id="x5:fcos:fcos_efficientnetb0_detect_512x512_bayese_nv12.bin",
            model_path="/tmp/private-fcos.bin",
        )
        self.assertEqual(selection.model_path, Path("/tmp/private-fcos.bin"))

    def test_real_source_post_process_matches_unified_for_quantized_fixture(self):
        from samples.vision.fcos.runtime.python.model_binding import bind_model, resolve_selection
        from samples.vision.fcos.runtime.python.fcos import FCOSTask

        selection = resolve_selection("x5", variant="efficientnetb0")
        metadata = metadata_for("efficientnetb0", per_channel=True)
        binding = bind_model(selection, metadata)
        raw = fixture_outputs()
        source_cls = _source_class()
        source = source_cls.__new__(source_cls)
        source.cfg = types.SimpleNamespace(
            strides=[8, 16, 32, 64, 128], classes_num=80, conf_thres=0.5,
            iou_thres=0.6, use_stride_scaling=True,
        )
        source.input_h = source.input_w = 512
        source.output_quants = metadata["output_quants"]
        source.model_name = "fcos_efficientnetb0"
        source.cls_output_names = [f"cls_{s}" for s in (8, 16, 32, 64, 128)]
        source.box_output_names = [f"box_{s}" for s in (8, 16, 32, 64, 128)]
        source.center_output_names = [f"center_{s}" for s in (8, 16, 32, 64, 128)]
        source.grids = {}
        for stride in source.cfg.strides:
            yv, xv = np.meshgrid(np.arange(512 // stride), np.arange(512 // stride))
            source.grids[stride] = (((np.stack((yv, xv), 2) + 0.5) * stride).reshape(-1, 2)).astype(np.float32)
        source_result = source.post_process({source.model_name: raw}, 512, 512)

        task = FCOSTask(lambda tensors: raw, binding)
        unified_result = task.post_process(raw, task.pre_process(np.zeros((512, 512, 3), dtype=np.uint8)).context)
        for expected, actual in zip(source_result, unified_result.as_tuple()):
            np.testing.assert_allclose(actual, expected, rtol=0, atol=0)

    def test_forward_preserves_raw_objects_and_predict_equals_explicit_stages(self):
        from samples.vision.fcos.runtime.python.fcos import FCOSTask
        from samples.vision.fcos.runtime.python.model_binding import bind_model, resolve_selection

        binding = bind_model(resolve_selection("x5"), metadata_for("efficientnetb0"))
        raw = fixture_outputs()
        seen = {}

        def runner(tensors):
            seen["input"] = tensors
            return raw

        task = FCOSTask(runner, binding)
        image = np.zeros((300, 500, 3), dtype=np.uint8)
        prepared = task.pre_process(image)
        outputs = task.forward(prepared)
        self.assertIs(outputs, raw)
        self.assertIs(outputs["cls_8"], raw["cls_8"])
        explicit = task.post_process(outputs, prepared.context)
        predicted = task.predict(image)
        self.assertEqual(explicit, predicted)
        self.assertEqual(prepared.context.original_shape, (300, 500))
        self.assertEqual(tuple(seen["input"][binding.input_names[0]].shape), (512 * 512 * 3 // 2,))

    def test_letterbox_post_process_restores_non_square_coordinates_from_context(self):
        from samples.vision.fcos.runtime.python.fcos import FCOSTask
        from samples.vision.fcos.runtime.python.model_binding import bind_model, resolve_selection

        binding = bind_model(resolve_selection("x5"), metadata_for("efficientnetb0"))
        raw = {name: value.copy() for name, value in fixture_outputs().items()}
        for value in raw.values():
            value[...] = -12
        raw["cls_8"][0, 20, 20, 0] = 10
        raw["center_8"][0, 20, 20, 0] = 10
        raw["box_8"][0, 20, 20, :] = (20, 20, 20, 20)
        task = FCOSTask(lambda tensors: raw, binding, resize_type=1)
        image = np.zeros((300, 500, 3), dtype=np.uint8)
        prepared = task.pre_process(image)
        result = task.post_process(raw, prepared.context)
        # prepare() resizes to (new_h=307,new_w=512) and pads top=102,left=0.
        expected = np.asarray([[(148 / 512) * 500, (148 - 102) / 307 * 300,
                                (180 / 512) * 500, (180 - 102) / 307 * 300]], dtype=np.float32)
        np.testing.assert_allclose(result.boxes, expected, rtol=0, atol=1e-6)

    def test_float_outputs_with_scale_follow_source_dequant_contract(self):
        from samples.vision.fcos.runtime.python.fcos import FCOSTask
        from samples.vision.fcos.runtime.python.model_binding import bind_model, resolve_selection

        binding = bind_model(resolve_selection("x5"), metadata_for("efficientnetb0", dtype="float32"))
        raw = {name: value.astype(np.float32) for name, value in fixture_outputs().items()}
        task = FCOSTask(lambda tensors: raw, binding)
        result = task.predict(np.zeros((512, 512, 3), dtype=np.uint8))
        # The fixed source dequantizes float arrays too when SCALE metadata is
        # present; 10.0 with scale .1 therefore yields a logit of 1.0.
        expected_score = 1.0 / (1.0 + np.exp(-1.0))
        np.testing.assert_allclose(result.scores[0], expected_score, rtol=0, atol=1e-6)

    def test_fixed_source_applies_scale_to_f32_outputs(self):
        _source_class()
        from utils.py_utils.postprocess import dequantize_tensor

        value = np.asarray([2.0], dtype=np.float32)
        np.testing.assert_array_equal(dequantize_tensor(value, Quant(0.5, 0)), np.asarray([1.0], dtype=np.float32))

    def test_binding_rejects_unknown_quant_descriptor(self):
        from samples.vision.fcos.runtime.python.model_binding import bind_model, resolve_selection

        metadata = metadata_for("efficientnetb0")
        metadata["output_quants"]["cls_8"] = None
        with self.assertRaises(ValueError):
            bind_model(resolve_selection("x5"), metadata)

    def test_binding_rejects_multi_model_and_non_array_outputs(self):
        from samples.vision.fcos.runtime.python.fcos import FCOSTask
        from samples.vision.fcos.runtime.python.model_binding import bind_model, resolve_selection

        metadata = metadata_for("efficientnetb0")
        metadata["model_names"] = (metadata["model_name"], "unrelated_model")
        with self.assertRaises(ValueError):
            bind_model(resolve_selection("x5"), metadata)

        binding = bind_model(resolve_selection("x5"), metadata_for("efficientnetb0"))
        raw = fixture_outputs()
        raw["cls_8"] = raw["cls_8"].tolist()
        with self.assertRaises(ValueError):
            FCOSTask(lambda tensors: raw, binding).predict(np.zeros((512, 512, 3), dtype=np.uint8))

    def test_post_process_rejects_context_with_wrong_geometry(self):
        from samples.vision.fcos.runtime.python.fcos import FCOSTask
        from samples.vision.fcos.runtime.python.model_binding import bind_model, resolve_selection
        from samples.vision.fcos.runtime.python.tensor_io import ImageContext

        binding = bind_model(resolve_selection("x5"), metadata_for("efficientnetb0"))
        task = FCOSTask(lambda tensors: fixture_outputs(), binding)
        bad_context = ImageContext((300, 500), (256, 256), 1, (307, 512), (102, 103, 0, 0))
        with self.assertRaises(ValueError):
            task.post_process(fixture_outputs(), bad_context)

    def test_download_alias_and_wrappers_are_portable(self):
        model_dir = SAMPLE / "model"
        alias = model_dir / "download_model.sh"
        self.assertTrue(alias.is_file())
        scripts = (alias, model_dir / "download.sh", model_dir / "fulldownload.sh", SAMPLE / "runtime" / "python" / "run.sh")
        for script in scripts:
            text = script.read_text(encoding="utf-8")
            entrypoint = "main.py" if script.name == "run.sh" else "download.py"
            self.assertIn(f'"${{SCRIPT_DIR}}/{entrypoint}"', text)
            self.assertIn("python3", text)
            self.assertNotIn(".venv/bin/python", text)

    def test_evaluator_writes_complete_success_and_difference_evidence(self):
        from samples.vision.fcos.evaluator.compare import run_comparison
        from samples.vision.fcos.runtime.python.fcos import FCOSTask
        from samples.vision.fcos.runtime.python.model_binding import bind_model, resolve_selection
        from samples._shared.runtime_meta import RuntimeMetadata

        metadata = metadata_for("efficientnetb0")
        base_raw = fixture_outputs()
        image = np.zeros((512, 512, 3), dtype=np.uint8)
        selection = resolve_selection(
            "x5",
            asset_id="x5:fcos:fcos_efficientnetb0_detect_512x512_bayese_nv12.bin",
            model_path=tempfile.gettempdir() + "/fcos-evaluator-fixture.bin",
        )
        Path(selection.model_path).write_bytes(b"fixture-model")

        def legacy_factory(selection, image, **kwargs):
            binding = bind_model(selection, metadata)
            task = FCOSTask(lambda tensors: base_raw, binding)
            prepared = task.pre_process(image)
            return {
                "metadata": RuntimeMetadata.from_mapping(metadata),
                "inputs": {key: value.copy() for key, value in prepared.tensors.items()},
                "raw": {key: value.copy() for key, value in base_raw.items()},
                "result": dict(zip(("boxes", "scores", "class_ids"), task.post_process(base_raw, prepared.context).as_tuple())),
            }

        class FakeRuntime:
            model_names = [metadata["model_name"]]
            input_names = {metadata["model_name"]: metadata["input_names"]}
            input_shapes = {metadata["model_name"]: metadata["input_shapes"]}
            output_names = {metadata["model_name"]: metadata["output_names"]}
            output_shapes = {metadata["model_name"]: metadata["output_shapes"]}
            input_dtypes = {metadata["model_name"]: metadata["input_dtypes"]}
            output_dtypes = {metadata["model_name"]: metadata["output_dtypes"]}
            output_quants = {metadata["model_name"]: metadata["output_quants"]}

            def __init__(self, path=None, changed=False):
                self.changed = changed

            def set_scheduling_params(self, **kwargs):
                self.scheduling = kwargs

            def run(self, inputs):
                values = {key: value.copy() for key, value in base_raw.items()}
                if self.changed:
                    values["cls_8"][0, 0, 0, 0] = -12
                return {metadata["model_name"]: values}

        fake_sdk = types.ModuleType("hbm_runtime")
        fake_sdk.QuantParams = Quant
        fake_sdk.HB_HBMRuntime = FakeRuntime

        with tempfile.TemporaryDirectory() as temp:
            evidence = Path(temp) / "success"
            old_sdk = sys.modules.get("hbm_runtime")
            sys.modules["hbm_runtime"] = fake_sdk
            try:
                with patch("samples.vision.fcos.evaluator.compare.require_execution_target", return_value="x5"):
                    passed = run_comparison(selection, image, SAMPLE / "test_data" / "bus.jpg", evidence,
                                            runtime_factory=lambda path: FakeRuntime())
            finally:
                if old_sdk is None:
                    sys.modules.pop("hbm_runtime", None)
                else:
                    sys.modules["hbm_runtime"] = old_sdk
            self.assertEqual(passed["return_code"], 0)
            self.assertTrue((evidence / "comparison.json").is_file())
            self.assertTrue((evidence / "source" / "raw" / "cls_8.npy").is_file())
            self.assertTrue((evidence / "unified" / "raw" / "cls_8.npy").is_file())
            self.assertTrue((evidence / "input.npy").is_file())

            difference = Path(temp) / "difference"
            old_sdk = sys.modules.get("hbm_runtime")
            sys.modules["hbm_runtime"] = fake_sdk
            try:
                with patch("samples.vision.fcos.evaluator.compare.require_execution_target", return_value="x5"):
                    failed = run_comparison(selection, image, SAMPLE / "test_data" / "bus.jpg", difference,
                                            runtime_factory=lambda path: FakeRuntime(changed=True))
            finally:
                if old_sdk is None:
                    sys.modules.pop("hbm_runtime", None)
                else:
                    sys.modules["hbm_runtime"] = old_sdk
            self.assertEqual(failed["return_code"], 1)
            self.assertFalse(failed["passed"])

            execution_error = Path(temp) / "execution-error"
            old_sdk = sys.modules.get("hbm_runtime")
            sys.modules["hbm_runtime"] = fake_sdk
            try:
                with patch("samples.vision.fcos.evaluator.compare.require_execution_target", return_value="x5"):
                    errored = run_comparison(selection, image, SAMPLE / "test_data" / "bus.jpg", execution_error,
                                             runtime_factory=lambda path: (_ for _ in ()).throw(RuntimeError("fake SDK failure")))
            finally:
                if old_sdk is None:
                    sys.modules.pop("hbm_runtime", None)
                else:
                    sys.modules["hbm_runtime"] = old_sdk
            self.assertEqual(errored["return_code"], 2)
            self.assertTrue((execution_error / "errors.json").is_file())
            self.assertIn("fake SDK failure", (execution_error / "errors.json").read_text(encoding="utf-8"))

    def test_runner_binds_injected_runtime_metadata_and_returns_raw_output(self):
        from samples.vision.fcos.runtime.python.model_binding import resolve_selection
        from samples.vision.fcos.runtime.python.model_runner import RuntimeModelRunner

        values = metadata_for("efficientnetb0")
        raw = fixture_outputs()

        class FakeRuntime:
            model_names = [values["model_name"]]
            input_names = {values["model_name"]: values["input_names"]}
            input_shapes = {values["model_name"]: values["input_shapes"]}
            input_dtypes = {values["model_name"]: values["input_dtypes"]}
            output_names = {values["model_name"]: values["output_names"]}
            output_shapes = {values["model_name"]: values["output_shapes"]}
            output_dtypes = {values["model_name"]: values["output_dtypes"]}
            output_quants = {values["model_name"]: values["output_quants"]}

            def run(self, inputs):
                return {values["model_name"]: raw}

        runner = RuntimeModelRunner(resolve_selection("x5"), runtime=FakeRuntime())
        binding = runner.load()
        tensors = {"input": np.zeros(512 * 512 * 3 // 2, dtype=np.uint8)}
        outputs = runner(tensors)
        self.assertEqual(binding.model_name, values["model_name"])
        self.assertIs(outputs["cls_8"], raw["cls_8"])

    def test_execution_identity_gate_precedes_default_sdk_factory(self):
        from samples.vision.fcos.runtime.python.model_binding import resolve_selection
        from samples.vision.fcos.runtime.python import model_runner

        runner = model_runner.RuntimeModelRunner(resolve_selection("x5"))
        with patch("samples._shared.platforms.require_execution_target", side_effect=ValueError("identity")), patch.object(model_runner, "_default_runtime_factory") as factory:
            with self.assertRaisesRegex(ValueError, "identity"):
                runner.load()
            factory.assert_not_called()

    def test_execution_requires_existing_selected_model_before_sdk_factory(self):
        from samples.vision.fcos.runtime.python.model_binding import resolve_selection
        from samples.vision.fcos.runtime.python import model_runner

        runner = model_runner.RuntimeModelRunner(resolve_selection("x5"))
        with patch("samples._shared.platforms.require_execution_target", return_value="x5"), patch.object(model_runner, "_default_runtime_factory") as factory:
            with self.assertRaises(ValueError):
                runner.load()
            factory.assert_not_called()

    def test_context_a_b_a_is_frozen_and_not_shared(self):
        from samples.vision.fcos.runtime.python.fcos import FCOSTask
        from samples.vision.fcos.runtime.python.model_binding import bind_model, resolve_selection

        binding = bind_model(resolve_selection("x5"), metadata_for("efficientnetb0"))
        task = FCOSTask(lambda _: fixture_outputs(), binding)
        a = task.pre_process(np.zeros((17, 31, 3), dtype=np.uint8))
        b = task.pre_process(np.zeros((29, 11, 3), dtype=np.uint8))
        a2 = task.pre_process(np.zeros((17, 31, 3), dtype=np.uint8))
        self.assertEqual(a.context.original_shape, (17, 31))
        self.assertEqual(b.context.original_shape, (29, 11))
        self.assertEqual(a2.context.original_shape, (17, 31))
        self.assertNotEqual(a.context, b.context)

    def test_letterbox_predict_and_explicit_stages_are_a_b_a_equivalent(self):
        from samples.vision.fcos.runtime.python.fcos import FCOSTask
        from samples.vision.fcos.runtime.python.model_binding import bind_model, resolve_selection

        binding = bind_model(resolve_selection("x5"), metadata_for("efficientnetb0"))
        raw = fixture_outputs()
        task = FCOSTask(lambda tensors: raw, binding, resize_type=1)
        image_a = np.zeros((300, 500, 3), dtype=np.uint8)
        image_b = np.full((500, 300, 3), 17, dtype=np.uint8)
        result_a = task.predict(image_a)
        prepared_b = task.pre_process(image_b)
        result_b = task.post_process(task.forward(prepared_b), prepared_b.context)
        result_a_again = task.predict(image_a)
        self.assertEqual(result_a, result_a_again)
        self.assertEqual(result_a.boxes.shape, (5, 4))
        self.assertEqual(result_b.boxes.shape, (5, 4))
        self.assertFalse(np.array_equal(result_a.boxes, result_b.boxes))

    def test_cli_help_list_and_dry_run_are_sdk_free(self):
        entry = SAMPLE / "runtime" / "python" / "main.py"
        for args in (("--help",), ("--list-models",), ("--dry-run", "--target", "x5")):
            result = subprocess.run([sys.executable, str(entry), *args], cwd="/tmp", capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("hbm_runtime", subprocess.run(
            [sys.executable, str(entry), "--help"], cwd="/tmp", capture_output=True, text=True
        ).stderr)

    def test_download_delegates_all_three_assets_without_network(self):
        from samples.vision.fcos.model import download

        calls = []
        with patch.object(download, "download_asset", side_effect=lambda asset, path: calls.append((asset.reference, Path(path))) or "digest"):
            result = download.download_target("x5", "/tmp/fcos-fixture", all_variants=True)
        self.assertEqual(len(result), 3)
        self.assertEqual(len(calls), 3)
        self.assertEqual([item[0] for item in calls], [
            "x5:fcos:fcos_efficientnetb0_detect_512x512_bayese_nv12.bin",
            "x5:fcos:fcos_efficientnetb2_detect_768x768_bayese_nv12.bin",
            "x5:fcos:fcos_efficientnetb3_detect_896x896_bayese_nv12.bin",
        ])


if __name__ == "__main__":
    unittest.main()
