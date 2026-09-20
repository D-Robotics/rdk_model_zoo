"""Host tests for the shared YOLO conversion workflow.

These tests exercise the public planning/configuration seam.  They do not need
OpenExplore, ONNX Runtime, or a board compiler; the actual toolchain is tested
by running the plan in its documented container.
"""

from argparse import Namespace
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import cv2
import numpy as np


SAMPLE = Path(__file__).resolve().parents[1]
CONVERSION = SAMPLE / "conversion"
if str(CONVERSION) not in sys.path:
    sys.path.insert(0, str(CONVERSION))

from workflow import (  # noqa: E402
    OnnxInput,
    make_conversion_plan,
    render_config,
    run_conversion,
    s_toolchain,
    x5_toolchain,
)


def options(**overrides):
    values = {
        "onnx": "models/yolo11n.onnx",
        "cal_images": "cal_images",
        "output_dir": ".",
        "quantized": "int8",
        "jobs": 16,
        "optimize_level": "O3",
        "cal_sample": True,
        "cal_sample_num": 20,
        "save_cache": False,
        "cal": ".calibration_data_temporary_folder",
        "ws": ".temporary_workspace",
    }
    values.update(overrides)
    return Namespace(**values)


class ConversionWorkflowTests(unittest.TestCase):
    def test_x5_plan_keeps_mapper_protocol_and_artifact_paths(self):
        plan = make_conversion_plan(
            options(),
            x5_toolchain(),
            OnnxInput("images", "tensor(float)", 640, 640),
            base_dir=Path("/work/sample"),
        )

        self.assertEqual(plan.calibration_suffix, ".rgbchw")
        self.assertFalse(plan.normalize_calibration)
        self.assertEqual(plan.artifact_path.name,
                         "yolo11n_bayese_640x640_nv12.bin")
        self.assertEqual(
            plan.compiler_command,
            ("hb_mapper", "makertbin", "--config", "config.yaml",
             "--model-type", "onnx"),
        )
        config = render_config(plan)
        self.assertIn('march: "bayes-e"', config)
        self.assertIn("cal_data_type: 'float32'", config)
        self.assertIn("optimization: set_Softmax_input_int8,set_Softmax_output_int8", config)
        self.assertNotIn("extra_params", config)

    def test_s_plan_uses_march_specific_hbm_and_normalized_npy(self):
        plan = make_conversion_plan(
            options(optimize_level="O2"),
            s_toolchain("nash-p"),
            OnnxInput("images", "tensor(float)", 640, 640),
            base_dir=Path("/work/sample"),
        )

        self.assertEqual(plan.calibration_suffix, ".npy")
        self.assertTrue(plan.normalize_calibration)
        self.assertEqual(plan.artifact_path.name,
                         "yolo11n_nashp_640x640_nv12.hbm")
        self.assertEqual(
            plan.compiler_command,
            ("hb_compile", "--config", "config.yaml"),
        )
        config = render_config(plan)
        self.assertIn('march: "nash-p"', config)
        self.assertIn("scale_value: 0.003921568627451", config)
        self.assertIn("extra_params:", config)
        self.assertIn("input_no_padding", config)

    def test_int16_is_encoded_by_each_target_protocol(self):
        x5 = make_conversion_plan(
            options(quantized="int16"),
            x5_toolchain(),
            OnnxInput("images", "tensor(float)", 640, 640),
            base_dir=Path("/work/sample"),
        )
        s = make_conversion_plan(
            options(quantized="int16", optimize_level="O2"),
            s_toolchain("nash-e"),
            OnnxInput("images", "tensor(float)", 640, 640),
            base_dir=Path("/work/sample"),
        )

        self.assertIn("set_all_nodes_int16", render_config(x5))
        self.assertIn('"all_node_type": "int16"', render_config(s))

    def test_s_toolchain_rejects_unknown_march(self):
        with self.assertRaises(ValueError):
            s_toolchain("nash-z")

    def test_failed_compilation_keeps_only_owned_workspace_for_diagnosis(self):
        class FakeSession:
            def get_inputs(self):
                return [SimpleNamespace(
                    name="images", type="tensor(float)", shape=[1, 3, 640, 640])]

        class FakeOrt:
            InferenceSession = lambda _path, providers: FakeSession()

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            onnx = root / "model.onnx"
            onnx.write_bytes(b"test")
            images = root / "images"
            images.mkdir()
            cv2.imwrite(str(images / "one.jpg"), np.zeros((4, 4, 3), np.uint8))
            workspace_parent = root / "workspaces"
            opts = options(
                onnx=str(onnx), cal_images=str(images),
                output_dir=str(root / "output"), ws=str(workspace_parent),
                cal_sample=False,
            )

            def fail_compile(command, cwd=None):
                if command == ("hb_mapper", "--version"):
                    return None
                raise RuntimeError("injected compiler failure")

            with self.assertRaises(RuntimeError):
                run_conversion(
                    opts, x5_toolchain(), ort_module=FakeOrt,
                    command_runner=fail_compile,
                )

            children = list(workspace_parent.iterdir())
            self.assertEqual(len(children), 1)
            self.assertTrue(children[0].name.startswith(".ultralytics-yolo-"))
            self.assertTrue((children[0] / "config.yaml").is_file())
            self.assertTrue(onnx.is_file())

    def test_default_workspace_parent_does_not_overlap_default_output(self):
        class FakeSession:
            def get_inputs(self):
                return [SimpleNamespace(
                    name="images", type="tensor(float)", shape=[1, 3, 640, 640])]

        class FakeOrt:
            InferenceSession = lambda _path, providers: FakeSession()

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            onnx = root / "model.onnx"
            onnx.write_bytes(b"test")
            images = root / "images"
            images.mkdir()
            cv2.imwrite(str(images / "one.jpg"), np.zeros((4, 4, 3), np.uint8))
            opts = options(
                onnx=str(onnx), cal_images=str(images),
                # This is the documented default shape: output is ONNX's
                # parent and --ws is a sibling parent containing temp children.
                output_dir=".", ws=".temporary_workspace", cal_sample=False,
            )

            def compile_success(command, cwd=None):
                if command == ("hb_mapper", "--version"):
                    return None
                self.assertEqual(command[0], "hb_mapper")
                artifact = Path(cwd) / "bpu_model_output" / (
                    "model_bayese_640x640_nv12.bin")
                artifact.parent.mkdir(parents=True, exist_ok=True)
                artifact.write_bytes(b"compiled")
                return None

            plan = run_conversion(
                opts, x5_toolchain(), base_dir=root, ort_module=FakeOrt,
                command_runner=compile_success,
            )
            self.assertTrue(plan.artifact_path.is_file())
            # The production path is resolved; on macOS the same temporary
            # directory is addressable as both /var/... and /private/var/...,
            # so compare resolved forms on every host.
            self.assertEqual(plan.artifact_path.parent, root.resolve())
            self.assertEqual(
                list((root / ".temporary_workspace").iterdir()), [])

    def test_workspace_parent_overlapping_calibration_pool_is_rejected(self):
        class FakeSession:
            def get_inputs(self):
                return [SimpleNamespace(
                    name="images", type="tensor(float)", shape=[1, 3, 640, 640])]

        class FakeOrt:
            InferenceSession = lambda _path, providers: FakeSession()

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            onnx = root / "model.onnx"
            onnx.write_bytes(b"test")
            images = root / "images"
            images.mkdir()
            cv2.imwrite(str(images / "one.jpg"), np.zeros((4, 4, 3), np.uint8))
            opts = options(
                onnx=str(onnx), cal_images=str(images),
                ws=str(images), output_dir=str(root / "output"),
            )
            with self.assertRaises(ValueError):
                run_conversion(
                    opts, x5_toolchain(), ort_module=FakeOrt,
                    command_runner=lambda command, cwd=None: None,
                )


if __name__ == "__main__":
    unittest.main()
