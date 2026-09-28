"""Focused host-side checks for YOLO26 calibration batch graph adaptation."""

from pathlib import Path
import importlib.util
import sys
import subprocess
import tempfile
import types
import unittest

import numpy as np

SAMPLE = Path(__file__).resolve().parents[1]
CONVERSION = SAMPLE / "conversion"
YOLO26_CONVERSION = CONVERSION / "yolo26"
sys.path.insert(0, str(YOLO26_CONVERSION))

from batch_flex import adapt_calibration_batch8


class _Dim:
    def __init__(self, value):
        self.dim_value = value
        self.dim_param = ""


class _ValueInfo:
    def __init__(self, shape):
        dims = types.SimpleNamespace(dim=[_Dim(value) for value in shape])
        self.type = types.SimpleNamespace(
            tensor_type=types.SimpleNamespace(shape=dims))


class _Tensor:
    def __init__(self, name, value):
        self.name = name
        self.value = np.array(value, copy=True)

    def CopyFrom(self, other):
        self.name = other.name
        self.value = np.array(other.value, copy=True)


class _Attribute:
    def __init__(self, name, tensor=None):
        self.name = name
        self.t = tensor
        self.ints = []


class _Node:
    def __init__(self, op_type, name, inputs=(), outputs=(), attributes=()):
        self.op_type = op_type
        self.name = name
        self.input = list(inputs)
        self.output = list(outputs)
        self.attribute = list(attributes)


class _FakeOnnx:
    def __init__(self, model):
        self.model = model
        self.numpy_helper = types.SimpleNamespace(
            to_array=lambda tensor: np.array(tensor.value, copy=True),
            from_array=lambda value, name="": _Tensor(name, value))
        self.checker = types.SimpleNamespace(check_model=self._check)

    def load(self, _path):
        return self.model

    def save(self, _model, path):
        Path(path).write_bytes(b"adapted")

    def _check(self, model):
        assert model is self.model


def _graph_model(include_attention=True):
    first_shape = _Tensor("shape_tensor", [1, 2, 3, 4])
    const = _Node(
        "Constant", "model.0.attn.shape", outputs=("shape0",),
        attributes=(_Attribute("value", first_shape),))
    first_reshape = _Node(
        "Reshape", "/model.0/attn/Reshape_0", ("x", "shape0"), ("a",))
    second_reshape = _Node(
        "Reshape", "/model.3.attn.Reshape_1", ("a", "shape1"), ("y",))
    unrelated = _Node(
        "Reshape", "/model.4/Reshape", ("x", "other_shape"), ("z",))
    graph = types.SimpleNamespace(
        input=[_ValueInfo((1, 3, 640, 640))],
        output=[_ValueInfo((1, 2, 3, 4))],
        node=[const, first_reshape, second_reshape, unrelated],
        initializer=[_Tensor("shape1", [1, 5, 6, 7]),
                     _Tensor("other_shape", [1, 8, 9, 10])],
    )
    if not include_attention:
        graph.node = [unrelated]
    return types.SimpleNamespace(graph=graph), first_shape


def _load_script(task):
    path = YOLO26_CONVERSION / f"export_yolo26_{task}_bpu.py"
    spec = importlib.util.spec_from_file_location(f"test_yolo26_{task}_export", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Yolo26BatchFlexTests(unittest.TestCase):
    def test_unified_dispatch_exposes_local_checkpoint_guard_for_every_task(self):
        dispatcher = CONVERSION / "export_monkey_patch.py"
        for task in ("detect", "cls", "seg", "pose", "obb"):
            with self.subTest(task=task):
                result = subprocess.run(
                    [sys.executable, str(dispatcher), "--family", "yolo26",
                     "--task", task, "--platform", "s100", "--help"],
                    capture_output=True, text=True, timeout=30)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn("--require-local", result.stdout)

    def test_patches_attention_constants_across_graph_variants_and_preserves_io(self):
        model, constant_shape = _graph_model()
        fake_onnx = _FakeOnnx(model)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.onnx"
            path.write_bytes(b"original")
            report = adapt_calibration_batch8(
                path, "pose", onnx_module=fake_onnx, numpy_module=np)

            self.assertEqual(report.status, "patched")
            self.assertEqual(len(report.patched_nodes), 2)
            np.testing.assert_array_equal(constant_shape.value, [-1, 2, 3, 4])
            np.testing.assert_array_equal(model.graph.initializer[0].value, [-1, 5, 6, 7])
            np.testing.assert_array_equal(model.graph.initializer[1].value, [1, 8, 9, 10])
            self.assertEqual(report.input_shapes, ((1, 3, 640, 640),))
            self.assertEqual(report.output_shapes, ((1, 2, 3, 4),))
            self.assertEqual(path.read_bytes(), b"adapted")

    def test_noop_fails_closed_and_leaves_original_file_untouched(self):
        model, _ = _graph_model(include_attention=False)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.onnx"
            path.write_bytes(b"original")
            with self.assertRaisesRegex(RuntimeError, "compatibility is unverified"):
                adapt_calibration_batch8(
                    path, "cls", onnx_module=_FakeOnnx(model), numpy_module=np)
            self.assertEqual(path.read_bytes(), b"original")

    def test_fused_none_heads_fall_back_to_complete_task_branches(self):
        class Tensor:
            def permute(self, *_axes):
                return self

        class Layer:
            def __init__(self):
                self.calls = 0

            def __call__(self, value):
                self.calls += 1
                return Tensor()

        def layers():
            return [Layer()]

        seg = _load_script("seg")
        class Segment:
            pass
        seg_head = Segment()
        seg_head.nl = 1
        seg_head.one2one_cv2 = seg_head.one2one_cv3 = seg_head.one2one_cv4 = None
        seg_head.cv2, seg_head.cv3, seg_head.cv4 = layers(), layers(), layers()
        seg_head.proto = lambda value: Tensor()
        seg_outputs = seg.bpu_segment_forward(seg_head, [Tensor()])
        self.assertEqual(len(seg_outputs), 4)

        pose = _load_script("pose")
        pose_head = types.SimpleNamespace(
            nl=1, one2one_cv2=None, one2one_cv3=None, one2one_cv4=None,
            cv2=layers(), cv3=layers(), cv4=layers())
        self.assertEqual(len(pose.bpu_pose_forward(pose_head, [Tensor()])), 3)

        pose26_head = types.SimpleNamespace(
            nl=1, one2one_cv2=layers(), one2one_cv3=layers(),
            one2one_cv4=layers(), one2one_cv4_kpts=None,
            cv2=layers(), cv3=layers(), cv4=layers(), cv4_kpts=layers())
        self.assertEqual(len(pose.bpu_pose_forward(pose26_head, [Tensor()])), 3)

        obb = _load_script("obb")
        obb_head = types.SimpleNamespace(
            nl=1, one2one_cv2=None, one2one_cv3=None, one2one_cv4=None,
            cv2=layers(), cv3=layers(), cv4=layers())
        self.assertEqual(len(obb.bpu_obb_forward(obb_head, [Tensor()])), 3)

    def test_segment_proto_dispatch_matches_head_api(self):
        seg = _load_script("seg")
        class Tensor:
            def permute(self, *_axes):
                return self

        class Layer:
            def __call__(self, _value):
                return Tensor()

        for head_type, expected_input in (("Segment", "single"), ("Segment26", "list")):
            calls = []
            Head = type(head_type, (), {})
            head = Head()
            head.nl = 1
            head.cv2, head.cv3, head.cv4 = [Layer()], [Layer()], [Layer()]
            head.proto = lambda value: calls.append(value) or Tensor()
            inputs = [Tensor()]
            seg.bpu_segment_forward(head, inputs)
            self.assertIs(calls[0], inputs[0] if expected_input == "single" else inputs)


if __name__ == "__main__":
    unittest.main()
