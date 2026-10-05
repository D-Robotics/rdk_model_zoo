"""Source preprocessing and corrected depth contracts on host fixtures."""

import ast
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import cv2
import numpy as np
from samples.vision.depth_anything_v2.runtime.python.model_binding import (
    resolve_selection,
    bind_model,
    list_available_assets,
)
from samples.vision.depth_anything_v2.runtime.python.model_runner import (
    RuntimeModelRunner,
)
from samples.vision.depth_anything_v2.runtime.python.depth_anything_v2 import (
    DepthAnythingV2Task,
    DepthResult,
)
from samples.vision.depth_anything_v2.runtime.python.visualization import (
    normalize_depth,
)

ROOT = Path(__file__).resolve().parents[4]


def metadata():
    return {
        "model_name": "depth",
        "model_names": ["depth"],
        "input_names": ["image"],
        "input_shapes": {"image": (1, 3, 518, 686)},
        "input_dtypes": {"image": "float32"},
        "output_names": ["pred"],
        "output_shapes": {"pred": (1, 518, 686)},
        "output_dtypes": {"pred": "float32"},
    }


def task(raw=None, mode=0):
    binding = bind_model(resolve_selection("s100"), metadata())
    return DepthAnythingV2Task(lambda tensors: raw.copy(), binding, resize_type=mode)


class DepthTests(unittest.TestCase):
    def test_exact_asset_and_target_rejection(self):
        self.assertEqual(len(list_available_assets()), 1)
        s = resolve_selection("s100")
        self.assertTrue(str(s.model_path).endswith("model/s100/depth_any.hbm"))
        for target in ("x5", "s100p", "s600"):
            self.assertEqual(list_available_assets(target), ())
            with self.assertRaises(ValueError):
                resolve_selection(target)
        with self.assertRaises(ValueError):
            resolve_selection("s100", model_path="/tmp/depth.hbm")
        with self.assertRaises(ValueError):
            resolve_selection("s100", asset_id="wrong")

    def test_metadata_binds_exact_source_geometry(self):
        for field, value in [
            ("input_shapes", {"image": (1, 3, 686, 518)}),
            ("input_dtypes", {"image": "uint8"}),
            ("output_shapes", {"pred": (1, 1, 518, 686)}),
            ("output_dtypes", {"pred": "int16"}),
        ]:
            m = metadata()
            m[field] = value
            with self.assertRaises(ValueError):
                bind_model(resolve_selection("s100"), m)

    def test_preprocess_matches_actual_source_zscore_not_imagenet(self):
        image = np.random.default_rng(12).integers(0, 256, (13, 17, 3), dtype=np.uint8)
        rgb = cv2.cvtColor(
            cv2.resize(image, (686, 518), interpolation=cv2.INTER_NEAREST),
            cv2.COLOR_BGR2RGB,
        )
        expected = (
            (
                (rgb - rgb.mean(axis=-1, keepdims=True))
                / np.sqrt(rgb.var(axis=-1, keepdims=True) + 1e-5)
            )
            .transpose(2, 0, 1)[None]
            .astype(np.float32)
        )
        result = task().pre_process(image)
        np.testing.assert_array_equal(result.tensors["image"], expected)
        self.assertEqual(
            (result.context.original_h, result.context.original_w), (13, 17)
        )

    def test_letterbox_preserves_source_input_and_crops_output(self):
        image = np.full((259, 686, 3), 17, np.uint8)
        t = task(mode=1)
        p = t.pre_process(image)
        self.assertEqual((p.context.top, p.context.bottom), (129, 130))
        np.testing.assert_array_equal(p.tensors["image"], 0)
        raw = np.zeros((1, 518, 686), np.float32)
        raw[:, 129:388] = 2
        result = t.post_process(raw, p.context)
        np.testing.assert_array_equal(result.depth_native, 2)

    def test_raw_forward_owned_runner_and_no_activation(self):
        raw = np.full((1, 518, 686), -3, np.float32)
        m = metadata()
        runtime = SimpleNamespace(
            **{k: v for k, v in m.items() if k != "model_name"},
            run=lambda data: {"depth": {"pred": raw}}
        )
        runner = RuntimeModelRunner(resolve_selection("s100"), runtime=runtime)
        b = runner.load()
        t = DepthAnythingV2Task(runner, b)
        p = t.pre_process(np.zeros((2, 3, 3), np.uint8))
        result = t.forward(p.tensors)
        np.testing.assert_array_equal(result, raw)
        self.assertFalse(np.shares_memory(result, raw))

    def test_context_is_per_call_and_validated(self):
        t = task()
        a = t.pre_process(np.zeros((2, 3, 3), np.uint8))
        t.pre_process(np.zeros((7, 9, 3), np.uint8))
        raw = np.ones((1, 518, 686), np.float32)
        self.assertEqual(t.post_process(raw, a.context).depth_native.shape, (2, 3))
        with self.assertRaises(ValueError):
            t.post_process(raw, replace(a.context, top=1))

    def test_invalid_inputs_and_outputs_fail(self):
        t = task()
        ctx = t.pre_process(np.zeros((2, 3, 3), np.uint8)).context
        for image in (
            np.zeros((0, 3, 3), np.uint8),
            np.zeros((2, 3), np.uint8),
            np.zeros((2, 3, 3), np.float32),
        ):
            with self.assertRaises(ValueError):
                t.pre_process(image)
        for raw in (
            np.zeros((518, 686), np.float32),
            np.zeros((1, 518, 686), np.int16),
            np.full((1, 518, 686), np.nan, np.float32),
        ):
            with self.assertRaises(ValueError):
                t.post_process(raw, ctx)
        with self.assertRaises(ValueError):
            task(mode=1).pre_process(np.zeros((1, 100000, 3), np.uint8))

    def test_nonuniform_half_pixel_restoration(self):
        # An affine plane has a closed-form bilinear result at half-pixel
        # coordinates. This checks restoration independently of cv2.resize.
        yy, xx = np.mgrid[:518, :686]
        raw = (yy * 2 + xx * 3).astype(np.float32)[None]
        t = task()
        ctx = t.pre_process(np.zeros((17, 23, 3), np.uint8)).context
        actual = t.post_process(raw, ctx).depth_native
        y = np.clip((np.arange(17) + 0.5) * 518 / 17 - 0.5, 0, 517)
        x = np.clip((np.arange(23) + 0.5) * 686 / 23 - 0.5, 0, 685)
        expected = 2 * y[:, None] + 3 * x[None, :]
        np.testing.assert_allclose(actual, expected, rtol=2e-7, atol=3e-4)

    def test_visualization_constant_and_full_range(self):
        np.testing.assert_array_equal(normalize_depth(np.ones((2, 3), np.float32)), 0)
        np.testing.assert_array_equal(
            normalize_depth(np.array([[-1, 0, 1]], np.float32)), [[0, 127, 255]]
        )
        with self.assertRaises(ValueError):
            normalize_depth(np.array([[np.inf]], np.float32))

    def test_stage_class_is_only_three_stages_and_predict(self):
        p = (
            ROOT
            / "samples/vision/depth_anything_v2/runtime/python/depth_anything_v2.py"
        )
        tree = ast.parse(p.read_text())
        c = next(
            n
            for n in tree.body
            if isinstance(n, ast.ClassDef) and n.name == "DepthAnythingV2Task"
        )
        self.assertEqual(
            {n.name for n in c.body if isinstance(n, ast.FunctionDef)},
            {
                "__init__",
                "preprocess",
                "infer",
                "postprocess",
                "predict",
                "pre_process",
                "forward",
                "post_process",
            },
        )

    def test_real_gate_before_sdk(self):
        with patch(
            "samples.vision.depth_anything_v2.runtime.python.model_runner.require_execution_target",
            side_effect=ValueError("wrong board"),
        ):
            with self.assertRaisesRegex(ValueError, "wrong board"):
                RuntimeModelRunner(resolve_selection("s100")).load()


class ReadableInterfaceTests(unittest.TestCase):
    """The canonical preprocess/infer/postprocess names drive predict."""

    def test_canonical_stages_exist_and_legacy_names_delegate(self):
        t = task(np.zeros((1, 518, 686), np.float32))
        image = np.zeros((13, 17, 3), np.uint8)
        for name in ("preprocess", "infer", "postprocess", "predict"):
            self.assertTrue(callable(getattr(t, name, None)), name)
        pn = t.preprocess(image)
        po = t.pre_process(image)
        self.assertEqual(vars(pn.context), vars(po.context))
        np.testing.assert_array_equal(pn.tensors["image"], po.tensors["image"])
        np.testing.assert_array_equal(t.infer(pn.tensors), t.forward(po.tensors))
        raw = np.zeros((1, 518, 686), np.float32)
        np.testing.assert_array_equal(
            t.postprocess(raw, pn.context).depth_native,
            t.post_process(raw, po.context).depth_native,
        )

    def test_predict_routes_through_canonical_stages_with_single_runner_call(self):
        raw = np.zeros((1, 518, 686), np.float32)
        calls = []

        def runner(tensors):
            calls.append(tensors)
            return raw.copy()

        binding = bind_model(resolve_selection("s100"), metadata())
        t = DepthAnythingV2Task(runner, binding)
        routed = []
        for canonical in ("preprocess", "infer", "postprocess"):
            original = getattr(t, canonical)

            def spy(*args, _original=original, _canonical=canonical, **kwargs):
                routed.append(_canonical)
                return _original(*args, **kwargs)

            setattr(t, canonical, spy)
        result = t.predict(np.zeros((13, 17, 3), np.uint8))
        self.assertEqual(routed, ["preprocess", "infer", "postprocess"])
        self.assertEqual(len(calls), 1)
        self.assertEqual(result.depth_native.shape, (13, 17))

    def test_legacy_stage_names_route_through_the_same_canonical_stages(self):
        t = task(np.zeros((1, 518, 686), np.float32))
        routed = []
        for canonical in ("preprocess", "infer", "postprocess"):
            original = getattr(t, canonical)

            def spy(*args, _original=original, _canonical=canonical, **kwargs):
                routed.append(_canonical)
                return _original(*args, **kwargs)

            setattr(t, canonical, spy)
        image = np.zeros((13, 17, 3), np.uint8)
        prepared = t.pre_process(image)
        t.post_process(t.forward(prepared.tensors), prepared.context)
        self.assertEqual(routed, ["preprocess", "infer", "postprocess"])

    def test_predict_equals_canonical_manual_composition_across_geometries(self):
        t = task(np.zeros((1, 518, 686), np.float32), mode=1)
        for image in (np.zeros((13, 17, 3), np.uint8), np.zeros((259, 686, 3), np.uint8)):
            prepared = t.preprocess(image)
            manual = t.postprocess(t.infer(prepared.tensors), prepared.context)
            np.testing.assert_array_equal(t.predict(image).depth_native, manual.depth_native)


class PredictionDetailsTests(unittest.TestCase):
    """Opt-in ``return_details`` archives this call's raw tensor in one pass."""

    def detailed_task(self):
        calls = []
        raw = np.arange(518 * 686, dtype=np.float32).reshape(1, 518, 686)

        def runner(tensors):
            calls.append(tensors)
            return raw.copy()

        binding = bind_model(resolve_selection("s100"), metadata())
        return DepthAnythingV2Task(runner, binding), calls

    def test_default_predict_still_returns_the_plain_result(self):
        t, calls = self.detailed_task()
        result = t.predict(np.zeros((13, 17, 3), np.uint8))
        self.assertIsInstance(result, DepthResult)
        self.assertEqual(len(calls), 1)

    def test_details_equal_explicit_stages_with_one_runner_call(self):
        t, calls = self.detailed_task()
        image = np.zeros((13, 17, 3), np.uint8)
        prepared = t.preprocess(image)
        raw = t.infer(prepared.tensors)
        expected = t.postprocess(raw, prepared.context)
        before = len(calls)
        details = t.predict(image, return_details=True)
        np.testing.assert_array_equal(details.result.depth_native, expected.depth_native)
        np.testing.assert_array_equal(details.raw, raw)
        np.testing.assert_array_equal(details.prepared.tensors["image"], prepared.tensors["image"])
        self.assertEqual(vars(details.prepared.context), vars(prepared.context))
        self.assertEqual(len(calls) - before, 1)

    def test_details_are_per_call_and_leave_no_state(self):
        t, _ = self.detailed_task()
        first = t.predict(np.zeros((13, 17, 3), np.uint8), return_details=True)
        second = t.predict(np.zeros((7, 9, 3), np.uint8), return_details=True)
        again = t.predict(np.zeros((13, 17, 3), np.uint8), return_details=True)
        self.assertEqual(first.result.depth_native.shape, (13, 17))
        self.assertEqual(second.result.depth_native.shape, (7, 9))
        self.assertEqual(again.result.depth_native.shape, (13, 17))
        self.assertEqual(set(vars(t)), {"runner", "binding", "resize_type"})


if __name__ == "__main__":
    unittest.main()
