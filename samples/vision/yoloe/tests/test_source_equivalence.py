"""Compare the fixed source's actual pure functions, with only SDK loading bypassed."""

import json
import hashlib
import importlib.util
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys
import unittest
from utils.py_utils.tests.legacy_platforms import legacy_path, legacy_tree  # noqa: E402
from unittest.mock import patch
import numpy as np
from test_runtime import fake_runtime, ROOT
from samples.vision.yoloe.runtime.python.model_binding import resolve_selection
from samples.vision.yoloe.runtime.python.model_runner import build_runner
from samples.vision.yoloe.runtime.python.yoloe import YOLOE, Config


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_source(target):
    for record in json.loads((Path(__file__).parent / "source-facts.json").read_text()):
        if (
            hashlib.sha256(legacy_path(record["path"][len("platforms/"):]).read_bytes()).hexdigest()
            != record["sha256"]
        ):
            raise AssertionError(f"Pinned source changed: {record['path']}")
    platform = legacy_tree(f"{target}/utils/py_utils").parents[0]
    relative = (
        "samples/vision/yoloe/runtime/python/yoloe_seg.py"
        if target == "x5"
        else "samples/vision/yoloe11_seg/runtime/python/yoloe11seg.py"
    )
    with patch.dict(sys.modules), patch.object(sys, "path", list(sys.path)):
        sdk = ModuleType("hbm_runtime")
        sdk.QuantParams = object
        sys.modules["hbm_runtime"] = sdk
        package = ModuleType("utils")
        package.__path__ = []
        sub = ModuleType("utils.py_utils")
        sub.__path__ = [str(legacy_tree(f"{target}/utils/py_utils"))]
        sys.modules["utils"] = package
        sys.modules["utils.py_utils"] = sub
        package.py_utils = sub
        for name in ("preprocess", "postprocess"):
            setattr(
                sub,
                name,
                load_file(
                    "utils.py_utils." + name, legacy_path(f"{target}/utils/py_utils/{name}.py")
                ),
            )
        module = load_file("legacy_yoloe_" + target, legacy_path(f"{target}/{relative}"))
    return module


class SourceEquivalence(unittest.TestCase):
    def test_x5_and_s11_numeric_results_match_source(self):
        for target in ("x5", "s100"):
            source = load_source("x5" if target == "x5" else "s")
            selection = resolve_selection(target)
            runtime, data = fake_runtime(selection)
            rng = np.random.default_rng(42)
            data["protos"][:] = rng.normal(size=data["protos"].shape).astype(np.float32)
            runner = build_runner(
                selection,
                runtime_loader=lambda: SimpleNamespace(HB_HBMRuntime=lambda p: runtime),
            )
            for morph in ((False,) if target == "x5" else (False, True)):
                task = YOLOE(selection, Config(do_morph=morph), runner=runner)
                cls = source.YOLOESeg if target == "x5" else source.YoloE11Seg
                legacy = cls.__new__(cls)
                legacy.cfg = (
                    source.YOLOESegConfig("unused")
                    if target == "x5"
                    else source.YoloE11SegConfig("unused", do_morph=morph)
                )
                legacy.model_name = "m"
                legacy.input_h = legacy.input_w = 640
                legacy.output_names = list(data)
                legacy._resize_type = 1
                legacy.weights_static = np.arange(16, dtype=np.float32)[None, None, :]
                legacy.output_quants = {
                    name: SimpleNamespace(quant_type="NONE") for name in data
                }
                image = np.zeros((320, 640, 3), np.uint8)
                result = task.predict(image)
                expected = legacy.post_process({"m": data}, 640, 320)
                np.testing.assert_allclose(result.boxes, expected[0], rtol=0, atol=1e-5)
                np.testing.assert_array_equal(result.scores, expected[1])
                np.testing.assert_array_equal(result.class_ids, expected[2])
                if target == "s100" and not morph:
                    # The source returns Lanczos output unchanged; overshoot
                    # values above 1 are part of the source output and are now
                    # preserved instead of renormalized.
                    self.assertGreater(
                        sum(np.count_nonzero(m > 1) for m in expected[3]), 0
                    )
                for actual, reference in zip(result.masks, expected[3]):
                    np.testing.assert_array_equal(actual, reference)
                self.assertEqual(len(result.masks), len(expected[3]))

    def test_x5_preprocess_matches_source_letterbox_and_stretch(self):
        source = load_source("x5")
        selection = resolve_selection("x5")
        runtime, _ = fake_runtime(selection)
        runner = build_runner(
            selection,
            runtime_loader=lambda: SimpleNamespace(HB_HBMRuntime=lambda p: runtime),
        )
        image = np.random.default_rng(12).integers(0, 256, (37, 59, 3), dtype=np.uint8)
        for resize_type in (0, 1):
            task = YOLOE(selection, Config(resize_type=resize_type), runner=runner)
            actual = task.pre_process(image).tensors["m"]["input"]
            pixels = source.pre_utils.resized_image(image, 640, 640, resize_type)
            y, uv = source.pre_utils.bgr_to_nv12_planes(pixels)
            expected = np.concatenate((y.reshape(-1), uv.reshape(-1)))
            np.testing.assert_array_equal(actual, expected)

    def test_x5_source_clamps_extreme_confidence_before_logit_conversion(self):
        source = load_source("x5")
        selection = resolve_selection("x5")
        runtime, data = fake_runtime(selection)
        data["cls_8"][0, 30, 30, 7] = -15
        data["cls_8"][0, 30, 31, 9] = -15
        runner = build_runner(
            selection,
            runtime_loader=lambda: SimpleNamespace(HB_HBMRuntime=lambda p: runtime),
        )
        task = YOLOE(selection, Config(score_thres=1e-8), runner=runner)
        expected = source.decode_seg_layer_dfl(
            data["box_8"],
            data["cls_8"],
            data["mces_8"],
            8,
            1e-8,
            4585,
            16,
            np.arange(16, dtype=np.float32)[None, None, :],
        )
        self.assertEqual(len(expected), 0)
        self.assertEqual(
            len(task.predict(np.zeros((320, 640, 3), np.uint8)).boxes), len(expected)
        )
