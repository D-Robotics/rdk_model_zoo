# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Numerical host tests for the CPU stages against captured TorchVision fixtures.

The ``tests/fixtures/*.npz`` files were captured from the pinned source
model (TorchVision 0.21.0, checkpoint sha256 907ea3f9...) on the standard
bus.jpg canvas; see conversion/README.md for the capture provenance.  Every
stage of the deployment CPU port — anchor table, RPN selection, RoIAlign,
PartB packing, final tail — must reproduce them.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from samples.vision.fasterrcnn.runtime.python import detect as D  # noqa: E402

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def load(name):
    with np.load(FIXTURES / name) as data:
        return {k: data[k] for k in data.files}


class AnchorTests(unittest.TestCase):
    def test_anchor_table_matches_torchvision_exactly(self):
        fixture = load("rpn_stage.npz")
        anchors = D.generate_anchors()
        self.assertEqual(anchors.shape, (3375, 4))
        self.assertEqual(anchors.dtype, np.float32)
        np.testing.assert_array_equal(anchors, fixture["anchors"])

    def test_anchor_level_counts(self):
        self.assertEqual(D.NUM_ANCHORS_PER_LEVEL, (1500, 1500, 375))
        table = D.generate_anchors()
        # TorchVision grid shifts are arange * stride (cell top-left), so the
        # first cell's 15 anchors are centered at (0, 0), not at (16, 16).
        first = table[:15]
        cx = 0.5 * (first[:, 0] + first[:, 2])
        cy = 0.5 * (first[:, 1] + first[:, 3])
        self.assertTrue(np.all(cx == 0.0) and np.all(cy == 0.0))


class RpnTests(unittest.TestCase):
    def test_rpn_select_matches_torchvision(self):
        fixture = load("rpn_stage.npz")
        proposals, n = D.rpn_select(fixture["rpn_obj"], fixture["rpn_deltas"],
                                    fixture["anchors"])
        self.assertEqual(n, int(fixture["n_valid"]))
        self.assertEqual(proposals.shape, (150, 4))
        np.testing.assert_allclose(proposals, fixture["proposals"], atol=1e-5)

    def test_all_filtered_input_yields_zero_proposals(self):
        anchors = D.generate_anchors()
        obj = np.full(3375, -50.0, dtype=np.float32)  # sigmoid ~ 0
        deltas = np.zeros((3375, 4), dtype=np.float32)
        proposals, n = D.rpn_select(obj, deltas, anchors)
        self.assertEqual(n, 0)
        np.testing.assert_array_equal(proposals, np.zeros((150, 4), dtype=np.float32))

    def test_non_finite_parta_output_rejected(self):
        anchors = D.generate_anchors()
        obj = np.zeros(3375, dtype=np.float32)
        obj[0] = np.nan
        with self.assertRaises(ValueError):
            D.rpn_select(obj, np.zeros((3375, 4), dtype=np.float32), anchors)


class RoIAlignTests(unittest.TestCase):
    def test_roialign_matches_torchvision(self):
        fixture = load("roialign_stage.npz")
        pooled = D.roi_align_feat0(fixture["feat0"], fixture["proposals"])
        rows = fixture["rows"]
        np.testing.assert_allclose(pooled[rows], fixture["pooled"], atol=1e-4, rtol=1e-4)

    def test_zero_area_roi_produces_finite_output(self):
        feat0 = np.arange(256 * 10 * 10, dtype=np.float32).reshape(256, 10, 10) / 1e3
        rois = np.array([[5.0, 5.0, 5.0, 5.0], [0.0, 0.0, 320.0, 320.0]], dtype=np.float32)
        pooled = D.roi_align_feat0(feat0, rois)
        self.assertEqual(pooled.shape, (2, 256, 7, 7))
        self.assertTrue(np.isfinite(pooled).all())

    def test_empty_rois(self):
        feat0 = np.zeros((256, 10, 10), dtype=np.float32)
        pooled = D.roi_align_feat0(feat0, np.zeros((0, 4), dtype=np.float32))
        self.assertEqual(pooled.shape, (0, 256, 7, 7))


class TailTests(unittest.TestCase):
    def test_tail_matches_torchvision(self):
        fixture = load("tail_stage.npz")
        boxes, scores, labels = D.tail(fixture["class_logits"], fixture["box_regression"],
                                       fixture["proposals"], int(fixture["n_valid"]))
        self.assertEqual(labels.tolist(), fixture["det_labels"].tolist())
        np.testing.assert_allclose(scores, fixture["det_scores"], atol=1e-6)
        np.testing.assert_allclose(boxes, fixture["det_boxes"], atol=1e-4)

    def test_tail_zero_valid_rows(self):
        logits = np.zeros((150, 91), dtype=np.float32)
        reg = np.zeros((150, 364), dtype=np.float32)
        prop = np.zeros((150, 4), dtype=np.float32)
        boxes, scores, labels = D.tail(logits, reg, prop, 0)
        self.assertEqual(boxes.shape, (0, 4))
        self.assertEqual(labels.shape, (0,))

    def test_background_never_returned_and_strict_threshold(self):
        # Scores clearly above the threshold survive with original ids; a
        # score clearly below it is dropped (the comparison itself is strict).
        logits = np.full((1, 91), -50.0, dtype=np.float32)
        logits[0, 0] = np.log(0.89)  # background absorbs the mass
        logits[0, 2] = np.log(0.06)   # class 2 at ~0.060 -> kept
        logits[0, 3] = np.log(0.049)  # class 3 at ~0.049 -> dropped
        reg = np.zeros((1, 364), dtype=np.float32)
        prop = np.array([[10.0, 10.0, 100.0, 100.0]], dtype=np.float32)
        boxes, scores, labels = D.tail(logits, reg, prop, 1)
        self.assertEqual(labels.tolist(), [2])
        self.assertGreater(float(scores[0]), 0.059)
        self.assertLess(float(scores[0]), 0.061)


class PackingTests(unittest.TestCase):
    def test_pack_bank_layout_roundtrip(self):
        rng = np.random.default_rng(7)
        pooled = rng.standard_normal((150, 256, 7, 7)).astype(np.float32)
        bank = D.pack_bank(pooled)
        self.assertEqual(bank.shape, (1, 256, 1050, 7))
        # Channel c, row r, bin (y,x): bank[0, c, r*7+y, x] == pooled[r, c, y, x]
        np.testing.assert_array_equal(bank[0, 5, 3 * 7 + 4, 6], pooled[3, 5, 4, 6])
        cls = rng.standard_normal((1, 91, 150, 1)).astype(np.float32)
        reg = rng.standard_normal((1, 364, 150, 1)).astype(np.float32)
        cls_out, reg_out = D.unpack_bank_outputs(cls, reg)
        self.assertEqual(cls_out.shape, (150, 91))
        self.assertEqual(reg_out.shape, (150, 364))
        np.testing.assert_array_equal(cls_out[42, 7], cls[0, 7, 42, 0])

    def test_pack_rejects_wrong_shape(self):
        with self.assertRaises(ValueError):
            D.pack_bank(np.zeros((10, 256, 7, 7), dtype=np.float32))


class GeometryTests(unittest.TestCase):
    def test_letterbox_square_and_wide(self):
        wide = D.letterbox_geometry(640, 161)
        self.assertEqual(wide.net_size, (320, 80))  # round(161*0.5)=80 (half-to-even)
        self.assertAlmostEqual(wide.scale, 320.0 / 640.0)
        tall = D.letterbox_geometry(161, 640)
        self.assertEqual(tall.net_size, (80, 320))
        odd = D.letterbox_geometry(319, 100)
        self.assertAlmostEqual(odd.scale, 320.0 / 319.0)
        self.assertEqual(odd.net_size, (320, 100))

    def test_canvas_to_original_mapping(self):
        # 1080x810 scales exactly (net 320x240, both axes rational); the
        # mapping must invert the achieved per-axis resize factors.
        context = D.letterbox_geometry(1080, 810)
        self.assertEqual(context.net_size, (320, 240))
        out = context.to_original(np.array([[100.0, 50.0, 200.0, 150.0]], dtype=np.float32))
        self.assertAlmostEqual(out[0, 0], 100.0 / (240.0 / 810.0), places=3)
        self.assertAlmostEqual(out[0, 1], 50.0 / (320.0 / 1080.0), places=3)
        self.assertAlmostEqual(out[0, 2], 200.0 / (240.0 / 810.0), places=3)
        self.assertAlmostEqual(out[0, 3], 150.0 / (320.0 / 1080.0), places=3)
        self.assertTrue(out[0, 2] < 810.0 and out[0, 3] < 1080.0)

    def test_invalid_geometry_rejected(self):
        with self.assertRaises(ValueError):
            D.letterbox_geometry(0, 100)


class NmsTests(unittest.TestCase):
    def test_suppression_is_strictly_greater(self):
        boxes = np.array([[0, 0, 10, 10], [0, 0, 10, 10]], dtype=np.float32)
        scores = np.array([0.9, 0.8], dtype=np.float32)
        keep = D.nms(boxes, scores, 1.0)  # IoU == 1.0 is NOT suppressed at 1.0
        self.assertEqual(keep.tolist(), [0, 1])
        keep = D.nms(boxes, scores, 0.99)
        self.assertEqual(keep.tolist(), [0])

    def test_no_plus_one_area_convention(self):
        # Adjacent unit boxes: continuous intersection is 0 either way.
        boxes = np.array([[0, 0, 1, 1], [1, 0, 2, 1]], dtype=np.float32)
        keep = D.nms(boxes, np.array([0.9, 0.8], dtype=np.float32), 0.0)
        self.assertEqual(keep.tolist(), [0, 1])

    def test_batched_groups_do_not_cross_suppress(self):
        boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11]], dtype=np.float32)
        scores = np.array([0.9, 0.8], dtype=np.float32)
        same = D.batched_nms(boxes, scores, np.array([0, 0]), 0.3)
        self.assertEqual(same.tolist(), [0])
        diff = D.batched_nms(boxes, scores, np.array([0, 1]), 0.3)
        self.assertEqual(sorted(diff.tolist()), [0, 1])


class DecodeTests(unittest.TestCase):
    def test_rpn_identity_weights_decode(self):
        anchors = np.array([[10.0, 10.0, 30.0, 40.0]], dtype=np.float32)
        deltas = np.zeros((1, 4), dtype=np.float32)
        np.testing.assert_array_equal(
            D.decode_deltas(anchors, deltas, D.RPN_BOX_WEIGHTS), anchors)

    def test_tail_decode_uses_predictor_weights(self):
        prop = np.array([[10.0, 10.0, 30.0, 40.0]], dtype=np.float32)
        deltas = np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32)
        out = D.decode_deltas(prop, deltas, D.BOX_WEIGHTS)[0]
        w, h = 20.0, 30.0
        cx, cy = 20.0, 25.0
        dw = min(0.3 / 5.0, D.BBOX_XFORM_CLIP)
        dh = min(0.4 / 5.0, D.BBOX_XFORM_CLIP)
        exp_w, exp_h = w * np.exp(dw), h * np.exp(dh)
        exp_cx, exp_cy = cx + 0.1 / 10.0 * w, cy + 0.2 / 10.0 * h
        np.testing.assert_allclose(
            out, [exp_cx - exp_w / 2, exp_cy - exp_h / 2,
                  exp_cx + exp_w / 2, exp_cy + exp_h / 2], rtol=1e-5)

    def test_xform_clip_limits_growth(self):
        prop = np.array([[0.0, 0.0, 10.0, 10.0]], dtype=np.float32)
        deltas = np.array([[0.0, 0.0, 100.0, 100.0]], dtype=np.float32)
        out = D.decode_deltas(prop, deltas, D.BOX_WEIGHTS)[0]
        max_side = 10.0 * np.exp(D.BBOX_XFORM_CLIP)
        self.assertLessEqual(out[2] - out[0], max_side + 1e-3)


class FixtureMetaTests(unittest.TestCase):
    def test_fixture_provenance_recorded(self):
        meta = json.loads((FIXTURES / "meta.json").read_text())
        self.assertEqual(meta["checkpoint_sha256"],
                         "907ea3f91ff92242bc1baea8049276a3e76bca48ce7560bd268cc029f37977b5")
        self.assertEqual(meta["torchvision"], "0.21.0+cpu")


if __name__ == "__main__":
    unittest.main()
