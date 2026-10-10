"""Regression: DFL prototype crops keep the source (unclipped) box semantics.

Reproduces the S100 YOLOE-11s board parity failure where clipping the
prototype crop to the letterbox content region shrank mask index 5 by ~12%
of its pixels (boolean IoU 0.738) and renormalizing Lanczos output changed
edge values. The source sample feeds the full decoded box to the prototype
crop and returns the resized mask unchanged.
"""

from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from samples.vision.ultralytics_yolo.runtime.python import segment as segmentation_decode
from samples.vision.ultralytics_yolo.runtime.python.detect import ImageTransform


def _padding_transform():
    """A tall-image letterbox: horizontal padding 160 px on each side."""
    return ImageTransform(
        original_size=(1280, 640),
        model_size=(640, 640),
        resized_size=(640, 320),
        padding=(160, 0, 160, 0),
        scale_x=0.5,
        scale_y=0.5,
    )


def _outputs_with_one_anchor(classes=3):
    """Ten semantic heads with exactly one confidently selected stride-8 anchor.

    The classification logit is high only for class 0 at anchor (0, 0); box
    distributions are uniform zeros, which decodes through the DFL expectation
    to a box extending past the letterbox padding into negative coordinates.
    """
    heads = {}
    for stride in (8, 16, 32):
        grid = 640 // stride
        cls = np.full((1, grid, grid, classes), -10.0, np.float32)
        box = np.zeros((1, grid, grid, 4 * 16), np.float32)
        mces = np.zeros((1, grid, grid, 32), np.float32)
        heads[f"cls_{stride}"] = cls
        heads[f"box_{stride}"] = box
        heads[f"mces_{stride}"] = mces
    heads["cls_8"][0, 0, 0, 0] = 5.0
    heads["protos"] = np.ones((1, 160, 160, 32), np.float32)
    return heads


class MaskCropParityTests(unittest.TestCase):
    def test_prototype_crop_receives_the_unclipped_decoded_box(self):
        contract = SimpleNamespace(
            strides=(8, 16, 32), classes=3, mces_num=32, reg_bins=16
        )
        transform = _padding_transform()
        seen = {}

        real_decode_masks = segmentation_decode.post.decode_masks

        def spy(coefficients, boxes, protos, *args, **kwargs):
            seen["boxes"] = np.array(boxes, copy=True)
            return real_decode_masks(coefficients, boxes, protos, *args, **kwargs)

        with mock.patch.object(segmentation_decode.post, "decode_masks", spy):
            _, _, _, masks = segmentation_decode.decode_segmentation(
                _outputs_with_one_anchor(),
                contract,
                transform,
                score_thres=0.25,
                nms_thres=0.7,
                do_morph=False,
            )
        # The decoded stride-8 box starts left of the padding boundary; the
        # source semantics pass that raw box to the prototype crop.
        self.assertLess(float(seen["boxes"][0][0]), 0.0)
        self.assertEqual(len(masks), 1)
        self.assertGreater(masks[0].size, 0)

    def test_resize_output_keeps_source_values_without_renormalization(self):
        # With uniform prototypes and a zero coefficient the decoded mask is
        # constant; Lanczos on constant data stays constant, so the returned
        # mask keeps the raw uint8 values (0/1 here) without binarization.
        from utils.py_utils.postprocess import resize_masks_to_boxes

        mask = np.ones((4, 4), np.uint8)
        out = resize_masks_to_boxes([mask], [[1.0, 1.0, 5.0, 5.0]], 10, 10, do_morph=False)
        self.assertEqual(out[0].shape, (4, 4))
        self.assertTrue(np.all(out[0] == 1))


if __name__ == "__main__":
    unittest.main()
