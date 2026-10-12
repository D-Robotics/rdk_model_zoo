# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Assembled-pipeline and runtime-binding regressions (independent review).

The stage fixtures in ``test_numerics.py`` pin each CPU stage against
captured TorchVision references; this module exercises what they cannot:
the *connected* pipeline (``preprocess -> infer -> middle -> pool ->
PartB -> tail -> to_original``) against fake SDK runtimes that replay the
captured PartA outputs and expose the live tensor metadata the binding
must validate, plus the RoIAlign malformed-ROI clamp transcribed
independently from the reference kernel.

Fake runtimes deliberately replay fixture tensors — never recompute them
from the implementation — so a stage-connection defect (e.g. feeding the
``[1,256,10,10]`` PartA output to a ``[C,H,W]`` stage) fails loudly here.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from samples.vision.fasterrcnn.runtime.python import detect as D  # noqa: E402

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def load(name):
    with np.load(FIXTURES / name) as data:
        return {k: data[k] for k in data.files}


# ---------------------------------------------------------------------------
# Independent RoIAlign reference: scalar loops transcribed from the
# TorchVision CPU kernel contract (unaligned, max(roi_w, 1) malformed-ROI
# clamp, [-1, H/W] validity window, border-clamped bilinear corners).
# ---------------------------------------------------------------------------

def _bilinear_reference(feat, y, x):
    h, w = feat.shape[1], feat.shape[2]
    if y < -1.0 or y > h or x < -1.0 or x > w:
        return np.zeros(feat.shape[0], dtype=np.float64)
    y = max(float(y), 0.0)
    x = max(float(x), 0.0)
    y_low, x_low = int(y), int(x)
    if y_low >= h - 1:
        y_low = y_high = h - 1
        y = float(h - 1)
    else:
        y_high = y_low + 1
    if x_low >= w - 1:
        x_low = x_high = w - 1
        x = float(w - 1)
    else:
        x_high = x_low + 1
    ly, lx = y - y_low, x - x_low
    return (feat[:, y_low, x_low] * (1 - ly) * (1 - lx)
            + feat[:, y_low, x_high] * (1 - ly) * lx
            + feat[:, y_high, x_low] * ly * (1 - lx)
            + feat[:, y_high, x_high] * ly * lx)


def roi_align_reference(feat, rois, spatial_scale=1.0 / 32.0,
                        output_size=7, sampling_ratio=2):
    """The reference kernel as plain scalar loops."""
    feat = np.asarray(feat, dtype=np.float64)
    n = len(rois)
    out = np.zeros((n, feat.shape[0], output_size, output_size), dtype=np.float64)
    for r in range(n):
        x1 = rois[r, 0] * spatial_scale
        y1 = rois[r, 1] * spatial_scale
        x2 = rois[r, 2] * spatial_scale
        y2 = rois[r, 3] * spatial_scale
        roi_w = max(x2 - x1, 1.0)
        roi_h = max(y2 - y1, 1.0)
        bin_w = roi_w / output_size
        bin_h = roi_h / output_size
        for iy in range(output_size):
            for ix in range(output_size):
                acc = np.zeros(feat.shape[0], dtype=np.float64)
                for fy in range(sampling_ratio):
                    for fx in range(sampling_ratio):
                        y = y1 + (iy + (fy + 0.5) / sampling_ratio) * bin_h
                        x = x1 + (ix + (fx + 0.5) / sampling_ratio) * bin_w
                        acc += _bilinear_reference(feat, y, x)
                out[r, :, iy, ix] = acc / (sampling_ratio * sampling_ratio)
    return out


class RoiAlignClampRegressions(unittest.TestCase):
    def test_reviewer_repro_first_bin(self):
        # feat f(y,x) = 10y + x (arange row-major), ROI [0,0,16,16] at
        # stride 32: 0.5 feature pixels wide -> forced 1x1 window.  Bin(0,0)
        # samples symmetric around (1/14, 1/14): value 11 * 1/14 = 11/14;
        # the unclamped implementation returned 11/28.
        feat = np.arange(100, dtype=np.float32).reshape(1, 10, 10)
        got = D.roi_align_feat0(feat, np.array([[0.0, 0.0, 16.0, 16.0]], dtype=np.float32))
        self.assertAlmostEqual(float(got[0, 0, 0, 0]), 11.0 / 14.0, places=6)
        self.assertAlmostEqual(11.0 / 14.0 / 2.0, 11.0 / 28.0, places=6)  # old defect value

    def test_full_tensors_match_reference_on_small_and_degenerate_rois(self):
        rng = np.random.default_rng(11)
        feat = rng.standard_normal((3, 8, 8)).astype(np.float32)
        rois = np.array([
            [0.0, 0.0, 16.0, 16.0],     # sub-pixel: clamp engages
            [10.0, 10.0, 10.0, 10.0],   # zero-area: clamp engages
            [5.0, 5.0, 31.9, 8.0],      # tall thin: clamp on x only
            [100.0, 100.0, 256.0, 256.0],
            [0.0, 0.0, 320.0, 320.0],
            [-20.0, -20.0, 340.0, 340.0],  # mostly outside: border/validity
        ], dtype=np.float32)
        got = D.roi_align_feat0(feat, rois)
        ref = roi_align_reference(feat, rois)
        # The implementation computes sample positions in float32 (as the
        # board-facing kernel does); this reference is pure float64, so a
        # few ulps of position rounding show up in the last digits.
        np.testing.assert_allclose(got, ref, atol=1e-5, rtol=1e-5)

    def test_random_rois_match_reference(self):
        rng = np.random.default_rng(12)
        feat = rng.standard_normal((4, 10, 10)).astype(np.float32)
        rois = rng.uniform(0.0, 320.0, size=(12, 4)).astype(np.float32)
        rois[:, 2:] = rois[:, :2] + np.abs(rois[:, 2:])  # x2 >= x1 etc.
        rois[:, 3] = np.maximum(rois[:, 3], rois[:, 1] + 3.0)
        rois[:, 2] = np.maximum(rois[:, 2], rois[:, 0] + 3.0)
        np.testing.assert_allclose(
            D.roi_align_feat0(feat, rois), roi_align_reference(feat, rois),
            atol=1e-5, rtol=1e-5)

    def test_zero_area_roi_is_finite_and_nonzero(self):
        # With the clamp a degenerate box pools a real 1x1 window; it must
        # not produce NaN or all-zero output.
        feat = np.arange(256 * 100, dtype=np.float32).reshape(256, 10, 10) / 97.0
        got = D.roi_align_feat0(feat, np.array([[160.0, 160.0, 160.0, 160.0]], dtype=np.float32))
        self.assertTrue(np.isfinite(got).all())
        self.assertGreater(float(np.abs(got).max()), 0.0)

    def test_wrong_ndim_rejected_with_shape(self):
        with self.assertRaises(ValueError):
            D.roi_align_feat0(np.zeros((1, 256, 10, 10), dtype=np.float32),
                              np.zeros((1, 4), dtype=np.float32))


# ---------------------------------------------------------------------------
# Fake board runtime: exposes the live SDK metadata surface and replays
# fixture tensors (or contract violations, on demand).
# ---------------------------------------------------------------------------

def _fake_shape(spec):
    """A concrete metadata shape for one contract spec (first alternative)."""
    if isinstance(spec, D._ShapeAlternatives):
        spec = spec.specs[0]
    return (spec,) if isinstance(spec, int) else tuple(spec)


class FakeSdkModel:
    """One loaded model replaying fixed outputs and recording its inputs."""

    def __init__(self, target, which, outputs, *,
                 input_names=None, input_dtypes=None, input_shapes=None,
                 output_names=None):
        contract = D.LIVE_TENSOR_CONTRACTS[target][which]
        self.model_names = ("model",)
        self.input_names = tuple(input_names if input_names is not None
                                 else contract["inputs"])
        self.input_shapes = dict(input_shapes if input_shapes is not None else {
            name: _fake_shape(spec)
            for name, (_, spec) in contract["inputs"].items()})
        self.input_dtypes = dict(input_dtypes if input_dtypes is not None else {
            name: dtypes[0] for name, (dtypes, _) in contract["inputs"].items()})
        # No quant metadata surface: the recorded contract binds dtypes and
        # shapes, and PartB quantization happens on chip inside the graph.
        self.output_names = tuple(output_names if output_names is not None
                                  else contract["outputs"])
        self.output_shapes = {name: shape for name, (_, shape) in contract["outputs"].items()
                              if name in self.output_names}
        self.output_dtypes = {name: dtype for name, (dtype, _) in contract["outputs"].items()
                              if name in self.output_names}
        self._outputs = dict(outputs)
        self.received: list[dict] = []

    def run(self, tensors):
        self.received.append({k: v for k, v in tensors.items()})
        return dict(self._outputs)


class FakeRuntimeSession:
    """A RuntimeSession stand-in: metadata lives on the wrapped object."""

    def __init__(self, sdk):
        self._sdk = sdk

    @property
    def runtime(self):
        return self._sdk

    def run(self, tensors):
        return self._sdk.run(tensors)


def fixture_pair(target):
    """Fake (parta, partb) models replaying the captured bus.jpg stages."""
    rpn = load("rpn_stage.npz")
    tail = load("tail_stage.npz")
    parta = FakeSdkModel(target, "parta", {
        "rpn_obj": rpn["rpn_obj"], "rpn_deltas": rpn["rpn_deltas"],
        "feat0": load("roialign_stage.npz")["feat0"].reshape(1, 256, 10, 10),
    })
    if target == "x5":
        partb = FakeSdkModel(
            target, "partb",
            {"class_logits": tail["class_logits"], "box_regression": tail["box_regression"]})
    else:
        cls = np.asarray(tail["class_logits"], dtype=np.float32).T.reshape(1, 91, 150, 1)
        reg = np.asarray(tail["box_regression"], dtype=np.float32).T.reshape(1, 364, 150, 1)
        partb = FakeSdkModel(target, "partb", {"cls_packed": cls, "reg_packed": reg})
    return parta, partb


def selection_for(target, **overrides):
    fields = dict(
        target=target,
        parta_path=f"/fake/{target}/parta",
        partb_path=f"/fake/{target}/partb",
        partb_layout=D.PARTB_LAYOUT_POOLED if target == "x5" else D.PARTB_LAYOUT_PACKED,
    )
    fields.update(overrides)
    return D.PairedSelection(**fields)


def task_with(parta, partb, selection):
    models = {str(selection.parta_path): parta, str(selection.partb_path): partb}
    runner = D.PairedRuntime(
        selection, runtime_factory=lambda path: FakeRuntimeSession(models[str(path)]))
    return D.FasterRCNNTask(runner=runner.load())


def bus_like_image():
    """A synthetic image with bus.jpg's 1080x810 geometry (PartA is a replay)."""
    rng = np.random.default_rng(3)
    return rng.integers(0, 255, size=(1080, 810, 3), dtype=np.uint8)


class AssembledPipelineRegressions(unittest.TestCase):
    """predict() must equal the staged calls and the captured tail fixture."""

    def _assert_fixture_tail(self, result, context):
        tail = load("tail_stage.npz")
        expected_boxes = context.to_original(tail["det_boxes"])
        np.testing.assert_allclose(result.boxes, expected_boxes, atol=1e-3)
        np.testing.assert_allclose(result.scores, tail["det_scores"], atol=1e-6)
        np.testing.assert_array_equal(result.labels, tail["det_labels"])

    def test_predict_matches_stages_and_fixture_x5(self):
        parta, partb = fixture_pair("x5")
        task = task_with(parta, partb, selection_for("x5"))
        image = bus_like_image()
        result = task.predict(image)
        # Staged calls must reproduce predict exactly.
        prepared = task.preprocess(image)
        self.assertEqual(prepared.context.net_size, (320, 240))
        outputs = task.infer(prepared)
        self.assertEqual(outputs["feat0"].shape, (1, 256, 10, 10))  # PartA layout
        staged = task.postprocess(outputs, prepared.context)
        self.assertEqual(result, staged)
        self._assert_fixture_tail(result, prepared.context)
        # The X5 PartB input is the float32 pooled batch: the graph
        # quantizes on chip with its recorded calibration threshold.
        received = partb.received[-1]
        self.assertEqual(set(received), {"pooled"})
        self.assertEqual(received["pooled"].dtype, np.float32)
        self.assertEqual(received["pooled"].shape, (150, 256, 7, 7))
        # And PartA consumed the packed NV12 bytes for this canvas.
        self.assertEqual(set(parta.received[-1]), {"images"})
        self.assertEqual(parta.received[-1]["images"].dtype, np.uint8)
        self.assertEqual(parta.received[-1]["images"].size, 320 * 320 * 3 // 2)

    def test_predict_matches_stages_and_fixture_s100(self):
        parta, partb = fixture_pair("s100")
        task = task_with(parta, partb, selection_for("s100"))
        image = bus_like_image()
        result = task.predict(image)
        prepared = task.preprocess(image)
        self.assertEqual(result, task.postprocess(task.infer(prepared), prepared.context))
        self._assert_fixture_tail(result, prepared.context)
        # Split NV12 planes on S.
        self.assertEqual(set(parta.received[-1]), {"images_y", "images_uv"})
        self.assertEqual(parta.received[-1]["images_y"].dtype, np.uint8)
        # The packed bank is float32 and carries exactly the pooled rows.
        bank = partb.received[-1]["bank"]
        self.assertEqual(bank.dtype, np.float32)
        self.assertEqual(bank.shape, (1, 256, 1050, 7))
        pooled_back = bank[0].reshape(256, 150, 7, 7).transpose(1, 0, 2, 3)
        outputs = task.infer(prepared)
        proposals, n, feat0 = task.middle(outputs)
        np.testing.assert_allclose(
            pooled_back[:n], task.pool(proposals, feat0, n)[:n], atol=1e-6)

    def test_pool_squeezes_parta_layout_and_zeroes_padding(self):
        feat4 = np.moveaxis(np.arange(256 * 100, dtype=np.float32).reshape(256, 10, 10), 0, 0)
        task_feat = feat4.reshape(1, 256, 10, 10)
        poison = np.full((D.NUM_PROPOSALS, 4), 777.0, dtype=np.float32)
        poison[:2] = np.array([[10.0, 10.0, 200.0, 200.0], [50.0, 50.0, 300.0, 250.0]],
                              dtype=np.float32)
        task = self._minimal_task("x5")
        pooled = task.pool(poison, task_feat, 2)
        self.assertEqual(pooled.shape, (150, 256, 7, 7))
        self.assertTrue((pooled[2:] == 0).all())          # padding rows zeroed
        np.testing.assert_allclose(
            pooled[:2], D.roi_align_feat0(feat4, poison[:2]), atol=1e-6)

    def test_pool_rejects_foreign_shapes(self):
        task = self._minimal_task("x5")
        for bad in ((10, 10), (256, 10, 11), (2, 256, 10, 10)):
            with self.subTest(shape=bad):
                with self.assertRaises(ValueError):
                    task.pool(np.zeros((150, 4), dtype=np.float32),
                              np.zeros(bad, dtype=np.float32), 1)

    def test_empty_proposals_skip_partb(self):
        parta, partb = fixture_pair("x5")
        dead = FakeSdkModel("x5", "parta", {
            "rpn_obj": np.full((3375, 1), -60.0, dtype=np.float32),
            "rpn_deltas": np.zeros((3375, 4), dtype=np.float32),
            "feat0": load("roialign_stage.npz")["feat0"].reshape(1, 256, 10, 10),
        })
        task = task_with(dead, partb, selection_for("x5"))
        result = task.predict(bus_like_image())
        self.assertEqual(len(result), 0)
        self.assertEqual(partb.received, [])               # PartB never ran

    def test_poisoned_padding_rows_never_leak(self):
        # PartB returns huge scores on rows >= n_valid; the tail must mask.
        tail = load("tail_stage.npz")
        n = int(tail["n_valid"])
        logits = np.asarray(tail["class_logits"], dtype=np.float32).copy()
        reg = np.asarray(tail["box_regression"], dtype=np.float32).copy()
        logits[n:] = 60.0            # certain "detections" on padding rows
        reg[n:] = 0.0
        parta, _ = fixture_pair("x5")
        partb = FakeSdkModel("x5", "partb", {"class_logits": logits, "box_regression": reg})
        task = task_with(parta, partb, selection_for("x5"))
        clean_a, clean_b = fixture_pair("x5")
        reference = task_with(clean_a, clean_b, selection_for("x5")).predict(bus_like_image())
        poisoned = task.predict(bus_like_image())
        self.assertEqual(reference, poisoned)

    def test_detection_result_owns_its_arrays(self):
        boxes = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
        scores = np.array([0.9], dtype=np.float32)
        labels = np.array([1], dtype=np.int64)
        result = D.DetectionResult(boxes, scores, labels)
        boxes[0, 0] = 90.0
        scores[0] = 0.1
        labels[0] = 7
        np.testing.assert_array_equal(result.boxes, [[1.0, 2.0, 3.0, 4.0]])
        np.testing.assert_allclose(result.scores, [0.9])
        np.testing.assert_array_equal(result.labels, [1])

    def test_next_predict_does_not_mutate_previous_result(self):
        parta, partb = fixture_pair("x5")
        task = task_with(parta, partb, selection_for("x5"))
        first = task.predict(bus_like_image())
        snapshot = (first.boxes.copy(), first.scores.copy(), first.labels.copy())
        task.predict(bus_like_image())
        np.testing.assert_array_equal(first.boxes, snapshot[0])
        np.testing.assert_array_equal(first.scores, snapshot[1])
        np.testing.assert_array_equal(first.labels, snapshot[2])

    @staticmethod
    def _minimal_task(target):
        parta, partb = fixture_pair(target)
        return task_with(parta, partb, selection_for(target))


class RuntimeBindingRegressions(unittest.TestCase):
    """Live-metadata reading and contract enforcement."""

    def test_x5_virtual_output_shapes_accepted(self):
        # Board repro (2026-10-12, x5-8g): the X5 packed-model spelling pads
        # the graph-declared shapes to 4D with trailing singleton dims —
        # rpn_obj (3375,1,1,1), rpn_deltas (3375,4,1,1), class_logits
        # (150,91,1,1), box_regression (150,364,1,1).  The flat buffers and
        # element counts are unchanged, so the binding must accept them.
        parta, partb = fixture_pair("x5")
        parta.output_shapes.update({
            "rpn_obj": (3375, 1, 1, 1),
            "rpn_deltas": (3375, 4, 1, 1),
        })
        partb.output_shapes.update({
            "class_logits": (150, 91, 1, 1),
            "box_regression": (150, 364, 1, 1),
        })
        task = task_with(parta, partb, selection_for("x5"))
        task.predict(bus_like_image())              # binding + run accepted

    def test_shape_mismatch_beyond_trailing_singletons_rejected(self):
        parta, partb = fixture_pair("x5")
        partb.output_shapes["class_logits"] = (150, 91, 2)   # not a pad of (150,91)
        with self.assertRaisesRegex(D.ContractError, "shape"):
            task_with(parta, partb, selection_for("x5"))

    def test_x5_pooled_input_must_be_float32(self):
        # The product PartB input is a float featuremap (makertbin
        # input_type_rt=featuremap; on-chip calibration quantizes), so an
        # int8-spelled input artifact is a different graph and must fail.
        parta, partb = fixture_pair("x5")
        partb.input_dtypes["pooled"] = "int8"
        with self.assertRaisesRegex(D.ContractError, "dtype"):
            task_with(parta, partb, selection_for("x5"))

    def test_metadata_reads_through_runtime_session_wrapper(self):
        # The tensor facts live on the underlying SDK object, not the
        # wrapper; the wrapper exposes only runtime/loaded/run/load.
        parta, partb = fixture_pair("x5")
        selection = selection_for("x5")
        runner = D.PairedRuntime(
            selection, runtime_factory=lambda path: FakeRuntimeSession(
                {str(selection.parta_path): parta, str(selection.partb_path): partb}[str(path)]))
        runner.load()      # metadata validation passed through the wrapper

    def test_sdk_run_envelope_unwrapped_by_model_name(self):
        # Board repro (2026-10-12, x5-8g): hbm_runtime.run returns
        # {model_name: {output_name: array}} (pybind dict[str,
        # dict[str, numpy.ndarray]]).  A single-model artifact must unwrap
        # to the inner output mapping before the output-name checks.
        parta, partb = fixture_pair("x5")
        parta.model_names = ("frcnn_mbv3_320_parta",)
        partb.model_names = ("frcnn_mbv3_320_partb",)
        run_parta, run_partb = parta.run, partb.run
        parta.run = lambda tensors: {"frcnn_mbv3_320_parta": run_parta(tensors)}
        partb.run = lambda tensors: {"frcnn_mbv3_320_partb": run_partb(tensors)}
        task = task_with(parta, partb, selection_for("x5"))
        result = task.predict(bus_like_image())
        self.assertGreaterEqual(len(result), 1)   # outputs were found by name

    def test_wrong_input_name_rejected(self):
        parta, partb = fixture_pair("s100")
        parta.input_names = ("frames_y", "frames_uv")
        with self.assertRaisesRegex(D.ContractError, "live inputs"):
            task_with(parta, partb, selection_for("s100"))

    def test_wrong_input_dtype_rejected(self):
        parta, partb = fixture_pair("s100")
        parta.input_dtypes = {"images_y": "float32", "images_uv": "uint8"}
        with self.assertRaisesRegex(D.ContractError, "dtype"):
            task_with(parta, partb, selection_for("s100"))

    def test_wrong_output_shape_rejected(self):
        parta, partb = fixture_pair("s100")
        partb.output_shapes["cls_packed"] = (1, 91, 150, 2)
        with self.assertRaisesRegex(D.ContractError, "shape"):
            task_with(parta, partb, selection_for("s100"))

    def test_x5_flat_input_byte_count_enforced(self):
        parta, partb = fixture_pair("x5")
        parta.input_shapes["images"] = (160 * 1024,)   # wrong byte count
        with self.assertRaisesRegex(D.ContractError, "images"):
            task_with(parta, partb, selection_for("x5"))

    def test_x5_image_layout_nv12_shape_accepted(self):
        # The X5 toolchain may report the packed NV12 input with the
        # (1,3,320,320) image layout instead of the flat byte buffer.
        parta, partb = fixture_pair("x5")
        parta.input_shapes["images"] = (1, 3, 320, 320)
        task = task_with(parta, partb, selection_for("x5"))
        task.predict(bus_like_image())          # binding accepted, no raise

    def test_nhwc_partb_shape_rejected(self):
        # Same element count, wrong declared layout: feeding NCHW bytes into
        # an NHWC-declared artifact would silently permute the input.
        parta, partb = fixture_pair("x5")
        partb.input_shapes["pooled"] = (150, 7, 7, 256)
        with self.assertRaisesRegex(D.ContractError, "pooled"):
            task_with(parta, partb, selection_for("x5"))

    def test_cleared_metadata_rejected(self):
        # Review repro: model_names present but every input/output name,
        # shape and dtype cleared — the old checker skipped absent facts.
        parta, partb = fixture_pair("s100")
        for attr, value in (("input_names", ()), ("input_dtypes", {}),
                            ("input_shapes", {}), ("output_names", ()),
                            ("output_dtypes", {}), ("output_shapes", {})):
            setattr(parta, attr, value)
        with self.assertRaisesRegex(D.ContractError, "live inputs"):
            task_with(parta, partb, selection_for("s100"))

    def test_missing_single_input_dtype_rejected(self):
        parta, partb = fixture_pair("s100")
        parta.input_dtypes.pop("images_uv")
        with self.assertRaisesRegex(D.ContractError, "images_uv.*dtype"):
            task_with(parta, partb, selection_for("s100"))

    def test_missing_single_output_shape_rejected(self):
        parta, partb = fixture_pair("s100")
        partb.output_shapes.pop("reg_packed")
        with self.assertRaisesRegex(D.ContractError, "reg_packed.*shape"):
            task_with(parta, partb, selection_for("s100"))

    def test_missing_live_metadata_rejected(self):
        class BareModel:               # no model_names surface at all
            def run(self, tensors):
                return {}
        selection = selection_for("x5")
        runner = D.PairedRuntime(selection, runtime_factory=lambda path: BareModel())
        with self.assertRaisesRegex(D.ContractError, "did not expose live tensor metadata"):
            runner.load()

    def test_partb_missing_outputs_raise_contract_error(self):
        parta, partb = fixture_pair("x5")
        partb._outputs.pop("box_regression")
        task = task_with(parta, partb, selection_for("x5"))
        with self.assertRaises(D.ContractError):
            task.predict(bus_like_image())


if __name__ == "__main__":
    unittest.main()
