"""Mock-Torch tests for the self-trained ResNet18 checkpoint export path.

These verify the export routine's calling contract only — checkpoint
loading (strict), fc class-count replacement, eval mode, fixed shapes and
export arguments — with fake torch/torchvision modules. The real export is
never executed here and torch is not installed in this environment; board
compilation and artifact claims stay separate.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest import mock

_SAMPLE = Path(__file__).resolve().parents[1]
_EXPORTER = _SAMPLE / "conversion" / "export_resnet18_onnx.py"


def _load_exporter():
    spec = importlib.util.spec_from_file_location("resnet_custom_export", _EXPORTER)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _FakeTorch:
    """Minimal torch double recording the calls the contract cares about."""

    def __init__(self, state):
        self.float32 = object()
        self.loaded_state = state
        self.load_calls = []
        self.export_calls = []
        self.zeros_calls = []
        self.__version__ = "0.0.0-mock"
        linear_calls = []

        class _Linear:
            def __init__(self, in_features, num_classes):
                linear_calls.append((in_features, num_classes))
                self.in_features = in_features
                self.out_features = num_classes

        class _NoGrad:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        self._linear_calls = linear_calls
        self.nn = types.SimpleNamespace(Linear=_Linear)
        self.no_grad = _NoGrad
        self.onnx = types.SimpleNamespace(export=self._record_export)

    def zeros(self, shape, dtype=None):
        self.zeros_calls.append((shape, dtype))
        return ("dummy", shape, dtype)

    def load(self, path, map_location=None):
        self.load_calls.append((path, map_location))
        return self.loaded_state

    def _record_export(self, model, dummy, path, **kwargs):
        self.export_calls.append((model, dummy, path, kwargs))


class _FakeModel:
    def __init__(self):
        self.fc = types.SimpleNamespace(in_features=512)
        self.load_state_dict_calls = []
        self.eval_called = False

    def load_state_dict(self, state, strict=None):
        self.load_state_dict_calls.append((state, strict))

    def eval(self):
        self.eval_called = True


class CustomExportTests(unittest.TestCase):
    def _run(self, state, *, num_classes=4, **kwargs):
        # A real (empty) checkpoint file: only its existence is checked on
        # the host path; the fake torch.load below never reads it.
        import tempfile
        checkpoint = Path(tempfile.mkdtemp()) / "ckpt.pth"
        checkpoint.write_bytes(b"")
        module = _load_exporter()
        torch = _FakeTorch(state)
        model = _FakeModel()
        resnet18_calls = []
        torchvision = types.ModuleType("torchvision")
        torchvision.__version__ = "0.0.0-mock"
        models = types.ModuleType("torchvision.models")

        def resnet18(weights=None):
            resnet18_calls.append(weights)
            return model

        models.resnet18 = resnet18
        models.ResNet18_Weights = types.SimpleNamespace(IMAGENET1K_V1=object())
        torchvision.models = models
        with mock.patch.dict(sys.modules, {"torch": torch, "torchvision": torchvision,
                                           "torchvision.models": models}):
            destination = module.export_resnet18(
                "out.onnx", checkpoint=checkpoint, num_classes=num_classes,
                check=False, **kwargs)
        return module, torch, model, resnet18_calls, destination, checkpoint

    def test_state_dict_export_call_contract(self):
        state = {"fc.weight": "w", "fc.bias": "b"}
        module, torch, model, resnet18_calls, destination, checkpoint = self._run(state)
        # No official weights are downloaded for a self-trained checkpoint.
        self.assertEqual(resnet18_calls, [None])
        # fc is replaced with the declared class count before loading.
        self.assertEqual(torch._linear_calls, [(512, 4)])
        self.assertEqual(model.fc.out_features, 4)
        # Strict state_dict load from a CPU-mapped checkpoint.
        self.assertEqual(torch.load_calls, [(str(checkpoint), "cpu")])
        self.assertEqual(model.load_state_dict_calls, [(state, True)])
        self.assertTrue(model.eval_called)
        # Fixed float32 NCHW dummy and pinned export arguments.
        self.assertEqual(torch.zeros_calls, [((1, 3, 224, 224), torch.float32)])
        (_, dummy, path, kwargs), = torch.export_calls
        self.assertEqual(dummy[1], (1, 3, 224, 224))
        self.assertIs(dummy[2], torch.float32)
        self.assertEqual(kwargs["input_names"], ["data"])
        self.assertEqual(kwargs["output_names"], ["output"])
        self.assertIsNone(kwargs["dynamic_axes"])
        self.assertFalse(kwargs["dynamo"])
        self.assertEqual(kwargs["opset_version"], 11)
        self.assertEqual(destination, Path("out.onnx"))

    def test_checkpoint_wrapper_state_dict_is_unwrapped(self):
        wrapped = {"state_dict": {"fc.weight": "w"}, "epoch": 3}
        _, torch, model, _, _, _ = self._run(wrapped)
        self.assertEqual(model.load_state_dict_calls,
                         [({"fc.weight": "w"}, True)])

    def test_missing_checkpoint_file_fails_with_path(self):
        module = _load_exporter()
        with self.assertRaises(FileNotFoundError) as raised:
            module.export_resnet18("out.onnx", checkpoint="/no/such/ckpt.pth",
                                   num_classes=4, check=False)
        self.assertIn("/no/such/ckpt.pth", str(raised.exception))

    def test_invalid_class_count_fails_before_torch_import(self):
        module = _load_exporter()
        for bad in (1, 0, -3):
            with self.subTest(num_classes=bad):
                with self.assertRaises(ValueError):
                    module.export_resnet18("out.onnx", checkpoint="ckpt.pth",
                                           num_classes=bad, check=False)

    def test_cli_rejects_checkpoint_with_official_weights(self):
        module = _load_exporter()
        args = module.build_parser().parse_args(
            ["--checkpoint", "ckpt.pth", "--num-classes", "4",
             "--weights", "IMAGENET1K_V1"])
        self.assertEqual(module.main([
            "--checkpoint", "ckpt.pth", "--num-classes", "4",
            "--weights", "IMAGENET1K_V1"]), 2)

    def test_cli_requires_num_classes_with_checkpoint(self):
        module = _load_exporter()
        self.assertEqual(
            module.main(["--checkpoint", "ckpt.pth"]), 2)


if __name__ == "__main__":
    unittest.main()
