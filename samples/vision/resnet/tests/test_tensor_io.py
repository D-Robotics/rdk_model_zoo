"""Host geometry and NV12 layout checks for the two observed protocols."""

from __future__ import annotations

import unittest

import cv2
import numpy as np


class TensorIoTests(unittest.TestCase):
    def test_letterbox_records_integer_geometry_and_padding(self):
        from samples.vision.resnet.runtime.python.tensor_io import resize_bgr

        image = np.zeros((100, 200, 3), dtype=np.uint8)
        _, transform = resize_bgr(
            image,
            224,
            224,
            resize_type=1,
            interpolation="nearest",
            letterbox_interpolation="nearest",
        )
        self.assertEqual(
            (
                transform.resized_height,
                transform.resized_width,
                transform.pad_top,
                transform.pad_bottom,
            ),
            (112, 224, 56, 56),
        )
        self.assertAlmostEqual(transform.scale_x, 1.12, places=8)
        self.assertAlmostEqual(transform.scale_y, 1.12, places=8)

    def test_letterbox_matches_legacy_helper_default_linear_resize(self):
        from samples.vision.resnet.runtime.python.tensor_io import resize_bgr

        # Textured, non-square input catches interpolation and integer-rounding
        # changes that a constant image would hide.
        rows, cols = np.indices((123, 456))
        image = np.stack(
            (
                (rows * 3 + cols * 5) % 256,
                (rows * 7 + cols * 11) % 256,
                (rows * 13 + cols * 17) % 256,
            ),
            axis=-1,
        ).astype(np.uint8)
        actual, _ = resize_bgr(image, 224, 224, resize_type=1)

        scale = min(224 / image.shape[0], 224 / image.shape[1])
        new_w = int(image.shape[1] * scale)
        new_h = int(image.shape[0] * scale)
        expected = cv2.resize(image, (new_w, new_h))
        expected = cv2.copyMakeBorder(
            expected,
            (224 - new_h) // 2,
            224 - new_h - (224 - new_h) // 2,
            (224 - new_w) // 2,
            224 - new_w - (224 - new_w) // 2,
            cv2.BORDER_CONSTANT,
            value=(127, 127, 127),
        )
        np.testing.assert_array_equal(actual, expected)

    def test_packed_nv12_is_y_then_interleaved_uv(self):
        from samples.vision.resnet.runtime.python.tensor_io import (
            bgr_to_nv12_planes,
            pack_nv12_single,
        )

        image = np.full((4, 4, 3), 128, dtype=np.uint8)
        y, uv = bgr_to_nv12_planes(image)
        packed = pack_nv12_single(
            image,
            input_width=4,
            input_height=4,
            resize_type=0,
            interpolation="nearest",
        )
        self.assertEqual(tuple(y.shape), (1, 4, 4, 1))
        self.assertEqual(tuple(uv.shape), (1, 2, 2, 2))
        self.assertEqual(packed.dtype, np.uint8)
        self.assertTrue(
            np.array_equal(
                packed.reshape(-1),
                np.concatenate((y.reshape(-1), uv.reshape(-1))),
            )
        )

    def test_bundled_white_wolf_keeps_legacy_letterbox_pixels(self):
        from pathlib import Path

        from samples.vision.resnet.runtime.python.tensor_io import resize_bgr

        image = cv2.imread(
            str(Path(__file__).parents[1] / "test_data" / "white_wolf.JPEG"),
            cv2.IMREAD_COLOR,
        )
        self.assertIsNotNone(image)
        actual, _ = resize_bgr(image, 224, 224, resize_type=1)
        height, width = image.shape[:2]
        scale = min(224 / height, 224 / width)
        new_w, new_h = int(width * scale), int(height * scale)
        expected = cv2.resize(image, (new_w, new_h))
        pad_w, pad_h = 224 - new_w, 224 - new_h
        expected = cv2.copyMakeBorder(
            expected,
            pad_h // 2,
            pad_h - pad_h // 2,
            pad_w // 2,
            pad_w - pad_w // 2,
            cv2.BORDER_CONSTANT,
            value=(127, 127, 127),
        )
        np.testing.assert_array_equal(actual, expected)

    def test_nv12_rejects_odd_target_geometry(self):
        from samples.vision.resnet.runtime.python.tensor_io import resize_bgr

        with self.assertRaises(ValueError):
            resize_bgr(np.zeros((4, 4, 3), dtype=np.uint8), 223, 224)


if __name__ == "__main__":
    unittest.main()
