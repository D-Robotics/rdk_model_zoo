"""Execute fixed S YOLO11 segmentation source against synthetic outputs."""

import unittest
from types import SimpleNamespace
import numpy as np
from test_segmentation_binding import fixture
import test_segmentation_binding as checks
from test_standalone_detection_source import source_task
from samples.vision.ultralytics_yolo.runtime.python.model_binding import (
    RuntimeMetadata,
    bind_model,
)
from samples.vision.ultralytics_yolo.runtime.python.model_runner import ModelRunner
from samples.vision.ultralytics_yolo.runtime.python.yolo_seg import YoloSeg


class SegmentationSource(unittest.TestCase):
    def test_source_float_scalar_and_symmetric_channel_masks_match(self):
        for mode in ("float", "scalar", "channel"):
            task, physical, quants = fixture(quantized=mode != "float")
            if mode == "scalar":
                for info in quants.values():
                    info.scale = np.array([0.25], np.float32)
            if mode == "channel":
                for info in quants.values():
                    info.zero_point.fill(
                        0
                    )  # Source discards scalar offsets in per-channel mode.
            metadata = RuntimeMetadata.from_runtime(task.model)
            binding = bind_model(task.binding.selection, metadata)
            task = YoloSeg(task.cfg, runner=ModelRunner(task.model, binding, metadata))
            if mode == "float":
                quants = {
                    name: SimpleNamespace(quant_type=SimpleNamespace(name="NONE"))
                    for name in physical
                }
            source = source_task(
                "yolo11_seg",
                "yolo11seg",
                "YoloV11Seg",
                "YoloV11SegConfig",
                quants,
                physical,
            )
            for resize in (0, 1):
                source.cfg.resize_type = task.cfg.resize_type = resize
                # Non-square sizes chosen with exact resize factors: unchanged geometry.
                for width, height in ((64, 64), (128, 64)):
                    for morph in (False, True):
                        with self.subTest(
                            mode=mode, resize=resize, size=(width, height), morph=morph
                        ):
                            source.cfg.do_morph = task.cfg.do_morph = morph
                            expected = source.post_process(
                                {"m": physical}, width, height
                            )
                            actual = task.post_process(task.forward({}), width, height)
                            checks.SegmentationBinding.assert_result_equal(
                                self, actual, expected
                            )
