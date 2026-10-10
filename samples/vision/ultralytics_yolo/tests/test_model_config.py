"""Configuration selection and visible model construction without a board SDK."""

import contextlib
import io
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

from samples.vision.ultralytics_yolo.runtime.python import main
from samples.vision.ultralytics_yolo.runtime.python import cli as yolo_dispatch
from samples.vision.ultralytics_yolo.runtime.python.cli import build_parser
from samples.vision.ultralytics_yolo.runtime.python.cli import resolve_platform


class ModelConfigTests(unittest.TestCase):
    def test_preparation_returns_task_class_and_config_without_loading_sdk(self):
        self.assertTrue(callable(getattr(yolo_dispatch, "prepare_runtime_model", None)))
        # These literal expectations catch wrong task routing and platform defaults.
        for target in ("x5", "s100", "s100p", "s600"):
            for task, name in (("detect", "YOLO26Detect"), ("cls", "YoloCls"),
                               ("seg", "YOLO26Seg"), ("pose", "YOLO26Pose"),
                               ("obb", "YOLO26OBB")):
                with self.subTest(target=target, task=task), patch.dict(
                    sys.modules, {"hbm_runtime": None}
                ):
                    args = build_parser().parse_args([
                        "--platform", target, "--family", "yolo26", "--task", task,
                        "--model-path", "/missing/model.hbm",
                    ])
                    profile = resolve_platform(target)
                    Model, config = yolo_dispatch.prepare_runtime_model(profile, args)
                    self.assertIsInstance(Model, type)
                    self.assertEqual(Model.__name__, name)
                    self.assertEqual(config.model_path, "/missing/model.hbm")
                    self.assertEqual(config.platform.key, target)
                    self.assertEqual(config.resize_type, 0 if task == "cls" else 1)
                    if task == "cls":
                        self.assertEqual(config.topk, 5)
                    else:
                        self.assertEqual(config.score_thres, 0.25)
                        self.assertEqual(config.strides, [8, 16, 32])
                        self.assertEqual(config.nms_thres, 0.7 if target == "x5" else 0.45)

    def test_dfl_overrides_reach_the_selected_config_and_yolo26_rejects_them(self):
        self.assertTrue(callable(getattr(yolo_dispatch, "prepare_runtime_model", None)))
        for task, option, value, attribute in (("detect", "--reg", "32", "reg"),
                                              ("pose", "--nkpt", "21", "nkpt"),
                                              ("seg", "--mc", "16", "mces_num")):
            args = build_parser().parse_args([
                "--platform", "x5", "--family", "yolo11", "--task", task,
                "--model-path", "/missing/custom.bin", option, value,
            ])
            _, config = yolo_dispatch.prepare_runtime_model(resolve_platform("x5"), args)
            self.assertEqual(getattr(config, attribute), int(value))
            args.family = "yolo26"
            with self.assertRaises(ValueError):
                yolo_dispatch.prepare_runtime_model(resolve_platform("x5"), args)

    def test_main_constructs_once_predicts_once_and_passes_scheduling(self):
        self.assertTrue(callable(getattr(yolo_dispatch, "prepare_runtime_model", None)))
        events = []
        image = np.full((4, 6, 3), 7, np.uint8)

        class RecordedModel:
            def __init__(self, config):
                events.append(("construct", config))
                self.model = object()

            def set_scheduling_params(self, **kwargs):
                events.append(("schedule", kwargs))

            def predict(self, value):
                self.assert_image = value
                events.append(("predict", value))
                return [(3, 0.9)]

        # Keep real argument-to-config preparation; only replace the board model.
        from samples.vision.ultralytics_yolo.runtime.python.classify import YoloClsConfig
        with patch("samples.vision.ultralytics_yolo.runtime.python.cli.get_task_types", return_value=(RecordedModel, YoloClsConfig)), \
             patch("samples.vision.ultralytics_yolo.runtime.python.main.require_execution_target"), patch("samples.vision.ultralytics_yolo.runtime.python.main.ensure_model"), \
             patch("utils.py_utils.file_io.load_image", return_value=image), \
             patch("utils.py_utils.inspect.print_model_info"), \
             contextlib.redirect_stdout(io.StringIO()) as output:
            status = main.main([
                "--platform", "x5", "--family", "yolo26", "--task", "cls",
                "--model-path", "/missing/custom.bin", "--priority", "5",
                "--bpu-cores", "1", "--topk", "3",
            ])
        self.assertEqual(status, 0)
        self.assertEqual([event[0] for event in events], ["construct", "schedule", "predict"])
        self.assertEqual(events[0][1].topk, 3)
        self.assertEqual(events[0][1].resize_type, 0)
        self.assertEqual(events[1][1], {"priority": 5, "bpu_cores": [1]})
        self.assertIs(events[2][1], image)
        self.assertIn("0.9", output.getvalue())
