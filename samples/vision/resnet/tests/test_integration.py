"""Tests for the canonical ResNet source/resource integration."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys
import unittest
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[4]
SAMPLE = ROOT / "samples" / "vision" / "resnet"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ResNetIntegrationTests(unittest.TestCase):
    def test_manifest_backed_downloader_resolves_and_downloads_exact_asset(self):
        from samples.vision.resnet.model import download

        observed = {}

        def fake_download(asset, destination):
            observed["asset"] = asset
            observed["destination"] = destination
            return "observed-sha"

        with mock.patch.object(download, "download_asset", fake_download):
            digest = download.download_target("s600", SAMPLE / "model")

        self.assertEqual(digest, "observed-sha")
        self.assertEqual(observed["asset"].reference,
                         "s:resnet18:s600/resnet18_224x224_nv12.hbm")
        self.assertEqual(
            observed["destination"],
            SAMPLE / "model" / "s600" / "resnet18_224x224_nv12.hbm",
        )

    def test_conversion_export_help_is_available_without_torch(self):
        script = SAMPLE / "conversion" / "export_resnet18_onnx.py"
        completed = subprocess.run(
            [sys.executable, str(script), "--help"],
            cwd=str(ROOT),
            text=True,
            capture_output=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--output", completed.stdout)
        self.assertIn("--opset", completed.stdout)

    def test_legacy_python_runtime_modules_are_sdk_free_import_shims(self):
        x5 = _load_module(
            ROOT / "platforms/x5/samples/vision/resnet/runtime/python/resnet.py",
            "legacy_x5_resnet",
        )
        s18 = _load_module(
            ROOT / "platforms/s/samples/vision/resnet18/runtime/python/resnet18.py",
            "legacy_s18_resnet",
        )
        self.assertTrue(callable(x5.ResNet))
        self.assertTrue(callable(s18.Resnet18))
        self.assertTrue(hasattr(x5, "ResNetConfig"))
        self.assertTrue(hasattr(s18, "Resnet18Config"))

    def test_legacy_commands_recognize_equals_form_overrides(self):
        x5 = _load_module(
            ROOT / "platforms/x5/samples/vision/resnet/runtime/python/main.py",
            "legacy_x5_main_options",
        )
        s18 = _load_module(
            ROOT / "platforms/s/samples/vision/resnet18/runtime/python/main.py",
            "legacy_s18_main_options",
        )
        self.assertTrue(x5._has_option(["--model-path=custom.bin"], "--model-path"))
        self.assertTrue(s18._has_option(["--label-file=custom.names"], "--label-file"))

    def test_x5_legacy_wrapper_preserves_nested_io_and_topk_result(self):
        x5 = _load_module(
            ROOT / "platforms/x5/samples/vision/resnet/runtime/python/resnet.py",
            "legacy_x5_runtime_behavior",
        )
        runtime = _FakeX5Runtime()
        model = x5.ResNet(x5.ResNetConfig("fixture.bin"), runtime=runtime)

        prepared = model.pre_process(np.zeros((19, 23, 3), dtype=np.uint8))
        outputs = model.forward(prepared)
        class_ids, probabilities, labels = model.post_process(outputs, topk=3)

        # H2: the adapter feeds the canonical flat packed-NV12 byte buffer
        # (same bytes as the former (1, 336, 224, 1) view).
        self.assertEqual(tuple(prepared[model.model_name][model.input_names[0]].shape),
                         (224 * 336,))
        self.assertEqual(runtime.calls[0][model.model_name][model.input_names[0]].dtype,
                         np.uint8)
        self.assertEqual(class_ids.tolist(), [42, 7, 0])
        self.assertEqual(len(probabilities), 3)
        self.assertEqual(labels, ["42", "7", "0"])

    def test_s18_legacy_wrapper_selects_s600_and_preserves_split_io(self):
        s18 = _load_module(
            ROOT / "platforms/s/samples/vision/resnet18/runtime/python/resnet18.py",
            "legacy_s18_runtime_behavior",
        )
        runtime = _FakeSRuntime()
        model = s18.Resnet18(
            s18.Resnet18Config("fixture.hbm", target="s600"), runtime=runtime
        )

        prepared = model.pre_process(np.zeros((17, 29, 3), dtype=np.uint8))
        outputs = model.forward(prepared)
        results = model.post_process(outputs, topk=2)

        self.assertEqual(model._binding.selection.target, "s600")
        self.assertEqual(
            set(prepared[model.model_name]), {model.input_names[0], model.input_names[1]}
        )
        self.assertEqual(
            tuple(prepared[model.model_name][model.input_names[0]].shape),
            (1, 224, 224, 1),
        )
        self.assertEqual(
            tuple(prepared[model.model_name][model.input_names[1]].shape),
            (1, 112, 112, 2),
        )
        self.assertEqual([class_id for class_id, _ in results], [42, 7])


class _FakeX5Runtime:
    model_names = ["resnet18_224x224_nv12"]
    input_names = {"resnet18_224x224_nv12": ["data"]}
    input_shapes = {"resnet18_224x224_nv12": {"data": (1, 3, 224, 224)}}
    input_dtypes = {"resnet18_224x224_nv12": {"data": "U8"}}
    output_names = {"resnet18_224x224_nv12": ["prob"]}
    output_shapes = {"resnet18_224x224_nv12": {"prob": (1, 1000, 1, 1)}}
    output_dtypes = {"resnet18_224x224_nv12": {"prob": "F32"}}

    def __init__(self):
        self.calls = []

    def run(self, payload):
        self.calls.append(payload)
        scores = np.full((1, 1000, 1, 1), -10.0, dtype=np.float32)
        scores[0, 42, 0, 0] = 8.0
        scores[0, 7, 0, 0] = 7.0
        scores[0, 0, 0, 0] = 6.0
        return {"resnet18_224x224_nv12": {"prob": scores}}


class _FakeSRuntime:
    model_names = ["resnet18_224x224_nv12"]
    input_names = {"resnet18_224x224_nv12": ["input_y", "input_uv"]}
    input_shapes = {
        "resnet18_224x224_nv12": {
            "input_y": (1, 224, 224, 1),
            "input_uv": (1, 112, 112, 2),
        }
    }
    input_dtypes = {
        "resnet18_224x224_nv12": {"input_y": "U8", "input_uv": "U8"}
    }
    output_names = {"resnet18_224x224_nv12": ["output"]}
    output_shapes = {"resnet18_224x224_nv12": {"output": (1, 1000)}}
    output_dtypes = {"resnet18_224x224_nv12": {"output": "F32"}}

    def __init__(self):
        self.calls = []

    def run(self, payload):
        self.calls.append(payload)
        scores = np.full((1, 1000), -10.0, dtype=np.float32)
        scores[0, 42] = 8.0
        scores[0, 7] = 7.0
        return {"resnet18_224x224_nv12": {"output": scores}}


if __name__ == "__main__":
    unittest.main()
