"""Behavioral tests for the readable YOLO detection entry (detect.py).

These tests pin the user-facing contract of ``YoloDetect``: the ``detect``
module keeps its public import surface, ``predict`` accepts image paths and
BGR arrays with per-call geometry, consecutive different-size images never
reuse a stale transform, one ``predict`` is exactly one model call, wrong
class channels and protocol mismatches fail with specific errors, and task
dispatch still selects the protocol-specific classes. All runners are
injected host fixtures; no board SDK is loaded and no board inference is
claimed.
"""

from pathlib import Path
import sys
import unittest
from unittest import mock

import numpy as np

from test_forward_purity import fixture

_RUNTIME_PYTHON = Path(__file__).resolve().parents[1] / "runtime" / "python"
if str(_RUNTIME_PYTHON) not in sys.path:
    sys.path.insert(0, str(_RUNTIME_PYTHON))

from samples.vision.ultralytics_yolo.runtime.python.detect import (  # noqa: E402
    YoloDetect,
    YoloDetectConfig,
)

_BUS = Path(__file__).resolve().parents[1] / "test_data" / "bus.jpg"


def _task(**config_kwargs):
    runner, _, _, contract = fixture()
    config = YoloDetectConfig(
        "fixture.bin", classes_num=1, contract=contract,
        nms_thres=0.45, **config_kwargs)
    return YoloDetect(config, runner=runner), runner


class YoloDetectImportSurfaceTests(unittest.TestCase):
    def test_detect_module_keeps_its_public_import_surface(self):
        import samples.vision.ultralytics_yolo.runtime.python.detect as detect_module

        self.assertIs(detect_module.YoloDetect, YoloDetect)
        self.assertIs(detect_module.YoloDetectConfig, YoloDetectConfig)
        from samples.vision.ultralytics_yolo.runtime.python.detect import (
            DetectionResult,
        )
        self.assertIs(detect_module.DetectionResult, DetectionResult)

    def test_stage_aliases_delegate_to_the_new_methods(self):
        task, _ = _task()
        image = np.zeros((7, 11, 3), np.uint8)

        via_alias = task.pre_process(image)
        via_new = task.preprocess(image)

        self.assertEqual(via_alias.transform, via_new.transform)
        for name in via_alias.tensors["m"]:
            np.testing.assert_array_equal(
                via_alias.tensors["m"][name], via_new.tensors["m"][name])


class YoloDetectFlowTests(unittest.TestCase):
    def test_predict_equals_the_explicit_three_steps(self):
        task, _ = _task()
        image = np.zeros((64, 64, 3), np.uint8)

        via_predict = task.predict(image)
        prepared = task.preprocess(image)
        outputs = task.infer(prepared)
        via_explicit = task.postprocess(outputs, transform=prepared.transform)

        for left, right in zip(via_predict, via_explicit):
            np.testing.assert_array_equal(left, right)

    def test_predict_executes_the_model_exactly_once(self):
        task, runner = _task()

        with mock.patch.object(runner.model, "run", wraps=runner.model.run) as run:
            task.predict(np.zeros((64, 64, 3), np.uint8))

        self.assertEqual(run.call_count, 1)

    def test_input_arrays_are_not_modified_in_place(self):
        task, _ = _task()
        image = np.arange(64 * 80 * 3, dtype=np.uint8).reshape(64, 80, 3)
        before = image.copy()

        task.predict(image)

        np.testing.assert_array_equal(image, before)

    def test_different_size_images_keep_their_own_geometry(self):
        task, _ = _task()
        wide = np.full((200, 300, 3), 30, np.uint8)
        small = np.full((40, 28, 3), 200, np.uint8)

        wide_result = task.predict(wide)
        small_result = task.predict(small)

        # Each result equals the explicit chain with its own transform —
        # a stale transform from the previous call would misplace boxes.
        for image, result in ((wide, wide_result), (small, small_result)):
            prepared = task.preprocess(image)
            outputs = task.infer(prepared)
            expected = task.postprocess(outputs, transform=prepared.transform)
            for left, right in zip(result, expected):
                np.testing.assert_array_equal(left, right)
            height, width = image.shape[:2]
            if len(result.boxes_xyxy):
                self.assertGreaterEqual(float(result.boxes_xyxy[:, 0].min()), 0.0)
                self.assertLessEqual(float(result.boxes_xyxy[:, 2].max()), width)
                self.assertGreaterEqual(float(result.boxes_xyxy[:, 1].min()), 0.0)
                self.assertLessEqual(float(result.boxes_xyxy[:, 3].max()), height)

        # The detection itself is non-empty for the fixture peak anchor and
        # differs between the two geometries' box scales.
        self.assertGreater(len(wide_result.scores), 0)
        self.assertGreater(len(small_result.scores), 0)


class YoloDetectSourceTests(unittest.TestCase):
    def test_predict_accepts_a_local_image_path(self):
        import cv2

        task, _ = _task()
        array = cv2.imread(str(_BUS), cv2.IMREAD_COLOR)
        expected = task.predict(array)

        from_string = task.predict(str(_BUS))
        from_path = task.predict(_BUS)

        for left, right in zip(expected, from_string):
            np.testing.assert_array_equal(left, right)
        for left, right in zip(expected, from_path):
            np.testing.assert_array_equal(left, right)

    def test_missing_image_path_error_names_the_path(self):
        task, _ = _task()
        missing = "/nonexistent/dir/image.jpg"

        with self.assertRaises(FileNotFoundError) as raised:
            task.predict(missing)

        self.assertIn(missing, str(raised.exception))

    def test_unsupported_source_type_is_a_concrete_type_error(self):
        task, _ = _task()

        with self.assertRaises(TypeError):
            task.predict(12345)


class YoloDetectProtocolErrorTests(unittest.TestCase):
    def test_wrong_class_channels_fail_with_a_specific_error(self):
        task, _ = _task()
        image = np.zeros((64, 64, 3), np.uint8)
        prepared = task.preprocess(image)

        # A 2-channel cls tensor cannot satisfy the declared 1-class DFL
        # contract; the decoder names the offending role.
        bad = np.zeros((1, 8, 8, 2), np.float32)
        outputs = {"cls_8": bad, "cls_16": np.zeros((1, 4, 4, 1), np.float32),
                   "cls_32": np.zeros((1, 2, 2, 1), np.float32)}
        for stride in (8, 16, 32):
            outputs[f"box_{stride}"] = np.zeros(
                (1, 64 // stride, 64 // stride, 64), np.float32)

        with self.assertRaises(ValueError) as raised:
            task.postprocess(outputs, transform=prepared.transform)

        self.assertIn("cls_8", str(raised.exception))

    def test_ltrb_class_rejects_a_dfl_contract(self):
        from samples.vision.ultralytics_yolo.runtime.python.detect import (
            YOLO26Detect,
            YOLO26DetectConfig,
        )

        runner, _, _, contract = fixture()  # DFL contract
        with self.assertRaisesRegex(ValueError, "LTRB"):
            YOLO26Detect(
                YOLO26DetectConfig("fixture.bin", classes_num=1, contract=contract),
                runner=runner,
            )

    def test_nms_free_class_rejects_an_nms_contract(self):
        from samples.vision.ultralytics_yolo.runtime.python.detect import (
            YoloV10Detect,
            YoloV10DetectConfig,
        )

        runner, _, _, contract = fixture()  # DFL contract with NMS
        with self.assertRaisesRegex(ValueError, "YOLOv10"):
            YoloV10Detect(
                YoloV10DetectConfig("fixture.bin", classes_num=1, contract=contract),
                runner=runner,
            )


class YoloDispatchTests(unittest.TestCase):
    def test_dispatch_still_selects_the_protocol_specific_classes(self):
        import importlib

        from samples.vision.ultralytics_yolo.runtime.python.cli import resolve_platform

        from samples.vision.ultralytics_yolo.runtime.python.detect import YoloDetect
        from samples.vision.ultralytics_yolo.runtime.python.cli import get_task_types

        dfl_cls, _ = get_task_types(resolve_platform("x5"), "yolo11", "detect")
        # Dispatch imports the readable detect.YoloDetect class directly.
        self.assertIs(dfl_cls, YoloDetect)

        ltrb_cls, _ = get_task_types(resolve_platform("s100"), "yolo26", "detect")
        self.assertIs(
            ltrb_cls,
            getattr(importlib.import_module("samples.vision.ultralytics_yolo.runtime.python.detect"), "YOLO26Detect"))

        nms_free_cls, _ = get_task_types(resolve_platform("s100"), "yolov10", "detect")
        v10_cls = getattr(
            importlib.import_module("samples.vision.ultralytics_yolo.runtime.python.detect"), "YoloV10Detect")
        self.assertIs(nms_free_cls, v10_cls)
        # The NMS-free adapter reuses the readable DFL implementation.
        self.assertTrue(issubclass(nms_free_cls, YoloDetect))


if __name__ == "__main__":
    unittest.main()
