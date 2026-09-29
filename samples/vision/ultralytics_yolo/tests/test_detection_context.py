"""Prepared geometry belongs to one image, including interleaved stage calls."""

import unittest
from dataclasses import FrozenInstanceError
import numpy as np

from test_forward_purity import fixture
from samples.vision.ultralytics_yolo.runtime.python.yolo_detect import (
    YoloDetect,
    YoloDetectConfig,
)
from samples.vision.ultralytics_yolo.runtime.python.yolo26_det import (
    YOLO26Detect,
    YOLO26DetectConfig,
)


class DetectionContext(unittest.TestCase):
    def tasks(self):
        for ltrb, Model, Config in [
            (False, YoloDetect, YoloDetectConfig),
            (True, YOLO26Detect, YOLO26DetectConfig),
        ]:
            for resize in (0, 1):
                runner, _, _, contract = fixture(sdk=True, ltrb=ltrb)
                yield Model(
                    Config(
                        "fixture.bin",
                        classes_num=1,
                        contract=contract,
                        resize_type=resize,
                        nms_thres=0.45,
                    ),
                    runner=runner,
                )

    def test_interleaved_images_have_frozen_context_and_predict_equivalence(self):
        a = np.zeros((3, 7, 3), np.uint8)
        b = np.zeros((11, 5, 3), np.uint8)
        for task in self.tasks():
            with self.subTest(task=type(task).__name__, resize=task.cfg.resize_type):
                pa = task.pre_process(a)
                pb = task.pre_process(b)
                self.assertEqual(pa.transform.original_size, (3, 7))
                self.assertEqual(pb.transform.original_size, (11, 5))
                with self.assertRaises(FrozenInstanceError):
                    pa.transform.resize_type = 9
                expected = task.post_process(
                    task.forward(pa.tensors), transform=pa.transform
                )
                for left, right in zip(expected, task.predict(a)):
                    np.testing.assert_array_equal(left, right)
                actual_b = task.post_process(
                    task.forward(pb.tensors), transform=pb.transform
                )
                for left, right in zip(actual_b, task.predict(b)):
                    np.testing.assert_array_equal(left, right)
                self.assertFalse(hasattr(task, "last_transform"))
                self.assertFalse(hasattr(task, "last_image_transform"))

    def test_legacy_tensor_mapping_and_explicit_dimensions_are_stateless(self):
        image = np.zeros((3, 7, 3), np.uint8)
        for task in self.tasks():
            prepared = task.pre_process(image)
            self.assertIs(prepared["m"], prepared.tensors["m"])
            old_tensors, old_transform = task.pre_process_with_transform(image)
            self.assertEqual(old_transform, prepared.transform)
            for name, value in old_tensors["m"].items():
                np.testing.assert_array_equal(value, prepared.tensors["m"][name])
            outputs = task.forward(prepared)
            explicit = task.post_process(outputs, 7, 3)
            modern = task.post_process(outputs, transform=prepared.transform)
            for left, right in zip(explicit, modern):
                np.testing.assert_array_equal(left, right)
            with self.assertRaises(ValueError):
                task.post_process(outputs)
            with self.assertRaises(ValueError):
                task.post_process(outputs, 99, 99, transform=prepared.transform)

    def test_invalid_bgr_inputs_fail_at_preprocessing_boundary(self):
        task = next(self.tasks())
        for image in [
            np.zeros((3, 7, 3), np.float32),
            np.zeros((0, 7, 3), np.uint8),
            np.zeros((3, 7), np.uint8),
        ]:
            with self.subTest(
                shape=image.shape, dtype=image.dtype
            ), self.assertRaisesRegex(ValueError, "BGR"):
                task.pre_process(image)
