"""Host tests for the audited OCR input protocols."""

from __future__ import annotations

import unittest

import cv2
import numpy as np


def _textured_image(height: int = 17, width: int = 29) -> np.ndarray:
    y, x = np.indices((height, width))
    return np.stack(
        (
            (x * 17 + y * 3) % 256,
            (x * 5 + y * 19 + 11) % 256,
            (x * 23 + y * 7 + 37) % 256,
        ),
        axis=-1,
    ).astype(np.uint8)


def _nv12_reference(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    height, width = image.shape[:2]
    yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV_I420).reshape(-1)
    area = height * width
    y = yuv[:area].reshape(height, width)[None, :, :, None]
    u = yuv[area : area + area // 4].reshape(height // 2, width // 2)
    v = yuv[area + area // 4 :].reshape(height // 2, width // 2)
    uv = np.stack((u, v), axis=-1)[None]
    return y, uv


class TensorIoTests(unittest.TestCase):
    def test_x5_detection_is_linear_resize_and_one_packed_nv12_tensor(self):
        from samples.vision.paddle_ocr.runtime.python.model_binding import resolve_pair
        from samples.vision.paddle_ocr.runtime.python.tensor_io import prepare_detection

        image = _textured_image()
        tensors = prepare_detection(image, resolve_pair("x5"))
        self.assertEqual(tuple(tensors), ("x",))
        packed = tensors["x"]
        self.assertEqual(packed.shape, (1, 960, 640, 1))
        self.assertEqual(packed.dtype, np.uint8)

        resized = cv2.resize(image, (640, 640), interpolation=cv2.INTER_LINEAR)
        y, uv = _nv12_reference(resized)
        expected = np.concatenate((y.reshape(-1), uv.reshape(-1))).reshape(
            (1, 960, 640, 1)
        )
        np.testing.assert_array_equal(packed, expected)

    def test_s100_detection_is_area_resize_and_split_nv12_planes(self):
        from samples.vision.paddle_ocr.runtime.python.model_binding import resolve_pair
        from samples.vision.paddle_ocr.runtime.python.tensor_io import prepare_detection

        image = _textured_image()
        tensors = prepare_detection(image, resolve_pair("s100"))
        self.assertEqual(set(tensors), {"x_y", "x_uv"})
        self.assertEqual(tensors["x_y"].shape, (1, 640, 640, 1))
        self.assertEqual(tensors["x_uv"].shape, (1, 320, 320, 2))
        self.assertEqual(tensors["x_y"].dtype, np.uint8)
        self.assertEqual(tensors["x_uv"].dtype, np.uint8)

        resized = cv2.resize(image, (640, 640), interpolation=cv2.INTER_AREA)
        expected_y, expected_uv = _nv12_reference(resized)
        np.testing.assert_array_equal(tensors["x_y"], expected_y)
        np.testing.assert_array_equal(tensors["x_uv"], expected_uv)

    def test_python_recognition_preparation_is_linear_rgb_f32_nchw_for_both_pairs(self):
        from samples.vision.paddle_ocr.runtime.python.model_binding import resolve_pair
        from samples.vision.paddle_ocr.runtime.python.tensor_io import prepare_recognition

        image = _textured_image(11, 37)
        expected_hwc = cv2.resize(image, (320, 48), interpolation=cv2.INTER_LINEAR)
        expected = (expected_hwc[:, :, ::-1].astype(np.float32) / 255.0).transpose(
            2, 0, 1
        )[None]
        for target in ("x5", "s100"):
            with self.subTest(target=target):
                tensors = prepare_recognition(image, resolve_pair(target))
                self.assertEqual(tuple(tensors), ("x",))
                self.assertEqual(tensors["x"].dtype, np.float32)
                np.testing.assert_array_equal(tensors["x"], expected)

    def test_input_validation_rejects_non_bgr_or_non_uint8_images(self):
        from samples.vision.paddle_ocr.runtime.python.model_binding import resolve_pair
        from samples.vision.paddle_ocr.runtime.python.tensor_io import (
            prepare_detection,
            prepare_recognition,
        )

        with self.assertRaises(ValueError):
            prepare_detection(np.zeros((8, 8), dtype=np.uint8), resolve_pair("x5"))
        with self.assertRaises(ValueError):
            prepare_recognition(
                np.zeros((8, 8, 3), dtype=np.float32), resolve_pair("s100")
            )


if __name__ == "__main__":
    unittest.main()
