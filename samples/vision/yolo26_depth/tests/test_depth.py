"""Host contracts and source numerical parity; no board claims."""

import ast
from dataclasses import replace
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from samples._shared.runtime_meta import MetadataMismatchError
from samples.vision.yolo26_depth.runtime.python.model_binding import (
    resolve_selection,
    list_available_assets,
    bind_model,
)
from samples.vision.yolo26_depth.runtime.python.model_runner import RuntimeModelRunner
from samples.vision.yolo26_depth.runtime.python.yolo26_depth import Yolo26DepthTask
from samples.vision.yolo26_depth.runtime.python.visualization import colorize_depth

ROOT = Path(__file__).resolve().parents[4]


def metadata(lite=False):
    return dict(
        model_name="depth",
        model_names=["depth"],
        input_names=["images"],
        input_shapes={"images": [1, 3, 768, 768]},
        input_dtypes={"images": "float32" if lite else "nv12"},
        output_names=["output0"],
        output_shapes={"output0": [1, 192, 192, 1]},
        output_dtypes={"output0": "float32"},
    )


def source_module():
    path = (
        ROOT / "platforms/s/samples/vision/yolo26_depth/runtime/python/yolo26_depth.py"
    )
    spec = importlib.util.spec_from_file_location("_depth_source_oracle", path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(
        sys.modules,
        {"hbm_runtime": SimpleNamespace(HB_HBMRuntime=None), spec.name: module},
    ):
        spec.loader.exec_module(module)
    return module


class DepthTests(unittest.TestCase):
    def setUp(self):
        self.image = np.arange(37 * 23 * 3, dtype=np.uint8).reshape(37, 23, 3)
        self.raw = np.linspace(-6, 6, 192 * 192, dtype=np.float32).reshape(
            1, 192, 192, 1
        )

    def task(self, target="x5", variant="n", runner=None):
        selection = resolve_selection(target, variant=variant)
        binding = bind_model(selection, metadata(selection.profile == "lite"))
        return Yolo26DepthTask(runner or (lambda tensors: self.raw.copy()), binding)

    def test_twenty_assets_default_and_profile_identity(self):
        self.assertEqual(len(list_available_assets()), 20)
        for target in ("x5", "s100", "s100p", "s600"):
            self.assertEqual(len(list_available_assets(target)), 5)
            self.assertEqual(resolve_selection(target).variant, "n")
            for variant in ("n", "s", "m", "l", "x"):
                s = resolve_selection(target, variant=variant)
                self.assertEqual(
                    s.profile,
                    "lite" if target != "x5" and variant in ("l", "x") else "nv12",
                )
                self.assertEqual(
                    resolve_selection("auto", asset_id=s.asset.reference), s
                )

    def test_selection_rejects_conflicts_and_forgery(self):
        s = resolve_selection("s100p", variant="l")
        for kw in (
            {"target": "s100", "asset_id": s.asset.reference},
            {"target": "s100p", "variant": "n", "asset_id": s.asset.reference},
            {"target": "x5", "model_path": "arbitrary.bin"},
            {"target": "s100", "variant": "z"},
        ):
            with self.assertRaises(ValueError):
                resolve_selection(**kw)
        with self.assertRaises(ValueError):
            bind_model(replace(s, profile="nv12"), metadata())

    def test_binding_rejects_wrong_geometry_and_boundary(self):
        selection = resolve_selection("s600", variant="l")
        for change in (
            {"input_shapes": {"images": [1, 3, 512, 512]}},
            {"input_dtypes": {"images": "nv12"}},
            {"output_shapes": {"output0": [1, 768, 768, 1]}},
            {"output_dtypes": {"output0": "int32"}},
            {"output_semantics": "log_depth"},
            {"model_names": ["depth", "extra"]},
        ):
            m = metadata(True)
            m.update(change)
            with self.assertRaises(MetadataMismatchError):
                bind_model(selection, m)

    def test_nv12_source_preprocess_and_postprocess(self):
        source = source_module()
        task = self.task("s100", "s")
        prepared = task.pre_process(self.image)
        padded, geometry = source.letterbox(self.image, 768)
        np.testing.assert_array_equal(
            prepared.tensors["images"], source.bgr_to_nv12(padded)
        )
        self.assertEqual(prepared.tensors["images"].shape, (768 * 768 * 3 // 2,))
        top, bottom, left, right = geometry
        depth = cv2.resize(
            np.exp(self.raw.squeeze()), (768, 768), interpolation=cv2.INTER_LINEAR
        )
        expected = cv2.resize(
            depth[top : 768 - bottom, left : 768 - right],
            (23, 37),
            interpolation=cv2.INTER_LINEAR,
        )
        result = task.post_process(self.raw, prepared.context)
        np.testing.assert_array_equal(result.depth_native, expected)
        np.testing.assert_array_equal(result.log_depth, self.raw.squeeze())
        self.assertIsNone(result.raw_logit)

    def test_lite_source_parity_both_constants(self):
        source = source_module()
        for v in ("l", "x"):
            task = self.task("s600", v)
            prepared = task.pre_process(self.image)
            np.testing.assert_array_equal(
                prepared.tensors["images"], source.featuremap(self.image, 768)
            )
            result = task.post_process(self.raw, prepared.context)
            a, b = source.LITE_CALIBRATION[v]
            expected_log = np.clip(self.raw.squeeze(), -4, 5) * a + b
            np.testing.assert_array_equal(result.log_depth, expected_log)
            np.testing.assert_array_equal(
                result.depth_native,
                cv2.resize(
                    np.exp(expected_log), (23, 37), interpolation=cv2.INTER_LINEAR
                ),
            )
            np.testing.assert_array_equal(result.raw_logit, self.raw.squeeze())

    def test_forward_is_raw_and_nchw_binding_keeps_same_depth(self):
        raw = self.raw.copy()
        task = self.task(runner=lambda tensors: raw)
        prepared = task.pre_process(self.image)
        self.assertIs(task.forward(prepared.tensors), raw)
        np.testing.assert_array_equal(raw, self.raw)
        m = metadata()
        m["output_shapes"] = {"output0": [1, 1, 192, 192]}
        binding = bind_model(resolve_selection("x5"), m)
        nchw = Yolo26DepthTask(lambda tensors: raw.reshape(1, 1, 192, 192), binding)
        a, b = task.predict(self.image), nchw.predict(self.image)
        np.testing.assert_array_equal(a.log_depth, b.log_depth)
        np.testing.assert_array_equal(a.depth_native, b.depth_native)
        self.assertEqual(a.context, b.context)
        self.assertIsNone(b.raw_logit)

    def test_source_evaluator_counterexample_not_recalibrated(self):
        task = self.task("s100", "s")
        prepared = task.pre_process(self.image)
        result = task.post_process(np.zeros_like(self.raw), prepared.context)
        np.testing.assert_array_equal(
            result.depth_native, np.ones((37, 23), np.float32)
        )

    def test_context_is_per_call_and_cross_profile_refused(self):
        task = self.task()
        a = task.pre_process(self.image)
        b = task.pre_process(self.image[:11, :7])
        self.assertEqual(
            task.post_process(self.raw, a.context).depth_native.shape, (37, 23)
        )
        self.assertEqual(
            task.post_process(self.raw, b.context).depth_native.shape, (11, 7)
        )
        lite = self.task("s600", "l").pre_process(self.image)
        with self.assertRaises(ValueError):
            task.post_process(self.raw, lite.context)
        with self.assertRaises(ValueError):
            task.post_process(self.raw, replace(a.context, top=768))

    def test_invalid_input_output_and_exp_overflow(self):
        task = self.task()
        for image in (
            np.zeros((1, 10000, 3), np.uint8),
            self.image.astype(np.float32),
            self.image[:, :, 0],
            np.empty((0, 2, 3), np.uint8),
        ):
            with self.assertRaises(ValueError):
                task.pre_process(image)
        ctx = task.pre_process(self.image).context
        for raw in (
            self.raw.astype(np.float64),
            self.raw.squeeze(),
            np.full_like(self.raw, np.nan),
            np.full_like(self.raw, 1000),
        ):
            with self.assertRaises(ValueError):
                task.post_process(raw, ctx)

    def test_no_mutable_last_transform_and_predict_equivalence(self):
        task = self.task()
        p = task.pre_process(self.image)
        np.testing.assert_array_equal(
            task.predict(self.image).depth_native,
            task.post_process(task.forward(p.tensors), p.context).depth_native,
        )
        self.assertEqual(set(vars(task)), {"runner", "binding"})
        result = task.predict(self.image)
        result.log_depth[:] = 123
        self.assertNotEqual(float(self.raw.flat[0]), 123)
        path = ROOT / "samples/vision/yolo26_depth/runtime/python/yolo26_depth.py"
        node = next(
            n
            for n in ast.parse(path.read_text()).body
            if isinstance(n, ast.ClassDef) and n.name == "Yolo26DepthTask"
        )
        self.assertEqual(
            {n.name for n in node.body if isinstance(n, ast.FunctionDef)},
            {"__init__", "pre_process", "forward", "post_process", "predict"},
        )

    def test_runner_transport_lazy_owned_and_exact_names(self):
        selection = resolve_selection("s100p", variant="x")
        m = metadata(True)
        calls = []
        runtime = SimpleNamespace(
            **{k: v for k, v in m.items() if k != "model_name"},
            run=lambda inputs: (
                calls.append(inputs) or {"depth": {"output0": self.raw}}
            )
        )
        runner = RuntimeModelRunner(selection, runtime=runtime)
        self.assertFalse(runner.loaded)
        binding = runner.load()
        task = Yolo26DepthTask(runner, binding)
        p = task.pre_process(self.image)
        raw = task.forward(p.tensors)
        self.assertEqual(set(calls[0]["depth"]), {"images"})
        raw[:] = 0
        self.assertNotEqual(float(self.raw.flat[0]), 0)
        with self.assertRaises(MetadataMismatchError):
            runner({"wrong": p.tensors["images"]})

    def test_render_source_parity_and_invalid_rejection(self):
        source = source_module()
        depth = np.exp(self.raw.squeeze())
        np.testing.assert_array_equal(
            colorize_depth(depth), source.colorize_depth(depth)
        )
        for bad in (
            np.full((2, 2), np.nan, np.float32),
            np.empty((0, 0), np.float32),
            np.ones((2, 2, 1), np.float32),
        ):
            with self.assertRaises(ValueError):
                colorize_depth(bad)


if __name__ == "__main__":
    unittest.main()
