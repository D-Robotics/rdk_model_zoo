"""Execute fixed S source postprocessors against host tensors, never SDK calls."""

import hashlib
import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

import numpy as np
from test_forward_purity import fixture
from samples.vision.ultralytics_yolo.runtime.python.yolo_detect import (
    YoloDetect,
    YoloDetectConfig,
)

ROOT = Path(__file__).resolve().parents[4]
# SOURCE_SHA is populated from Git source pin 380e1a2 at test authoring, not at run time.
SOURCE_SHA = {
    'samples/vision/yolo11_seg/runtime/python/yolo11seg.py': 'b1abd70ab03467799d4f2253107c8c251fbd2003c081c19e604df0bed9d77967',
    "samples/vision/yolo11/runtime/python/yolo11.py": "19cbf4e3283259b14db8ea1b93aa3127fdd1bda46eb9d522f58f5cefe1f0909b",
    "samples/vision/yolov13_imoonlab/runtime/python/yolov13.py": "37d39d9b339a71f2d3a2f06590f863f5d2e7508a1014424136ae6b37146b77c5",
    "utils/py_utils/nn_math.py": "c8f431da01b644d26285a013474b775744c0e330bf39ea04477e8542fe1e08ff",
    "utils/py_utils/postprocess.py": "fdd17463ad08f4d0cd54450a2ed36953c10531c68e9060bbb9b7fe3e73587f48",
    "utils/py_utils/preprocess.py": "63c2bd0c64fb7d7f397e00d8b631d0437aef8817759f46e5b4633db5b8eb06f4",
}


def load_source(path, name):
    archive = ROOT / "platforms/s" / path
    if hashlib.sha256(archive.read_bytes()).hexdigest() != SOURCE_SHA[path]:
        raise AssertionError(f"Fixed source bytes changed: {path}")
    spec = importlib.util.spec_from_file_location(name, archive)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def source_task(sample, filename, class_name, config_name, quants, names):
    packages = {
        name: types.ModuleType(name)
        for name in ["utils", "utils.py_utils", "hbm_runtime"]
    }
    packages["utils"].__path__ = []
    packages["utils.py_utils"].__path__ = []
    packages["utils"].py_utils = packages["utils.py_utils"]
    packages["hbm_runtime"].QuantParams = object
    with patch.dict(sys.modules, packages), patch.object(sys, "path", list(sys.path)):
        for helper in ["nn_math", "preprocess", "postprocess"]:
            module = load_source(
                f"utils/py_utils/{helper}.py", f"utils.py_utils.{helper}"
            )
            setattr(packages["utils.py_utils"], helper, module)
        module = load_source(
            f"samples/vision/{sample}/runtime/python/{filename}.py", "fixed_source_yolo"
        )
        Model = getattr(module, class_name)
        model = Model.__new__(Model)  # Deliberately bypass the eager SDK loader.
        model.cfg = getattr(module, config_name)(
            "fixture.hbm", classes_num=1, anchor_sizes=[8, 4, 2]
        )
        model.model_name = "m"
        model.output_names = list(names)
        model.output_quants = quants
        model.input_h = model.input_w = 64
        model.weights_static = np.arange(16, dtype=np.float32)[None, None, :]
        return model


class StandaloneSourceParity(unittest.TestCase):
    def test_yolo11_source_scalar_and_symmetric_channel_dequant_match(self):
        for channels in (False, True):
            runner, physical, quants, contract = fixture(sdk=True, channels=channels)
            if channels:
                # The source discards scalar nonzero offsets for channel SCALE;
                # symmetric descriptors are the legitimate unchanged comparison.
                runner, physical, quants, contract = self.zero_offset_fixture()
            task = YoloDetect(
                YoloDetectConfig(
                    "fixture.bin", classes_num=1, contract=contract, nms_thres=0.45
                ),
                runner=runner,
            )
            source = source_task(
                "yolo11", "yolo11", "YoloV11", "YOLOv11Config", quants, physical
            )
            expected = source.post_process({"m": physical}, 64, 64)
            actual = task.post_process(task.forward({}), 64, 64)
            for a, b in zip(actual, expected):
                np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-5)

    @staticmethod
    def zero_offset_fixture():
        from samples.vision.ultralytics_yolo.runtime.python.model_binding import (
            RuntimeMetadata,
            bind_model,
        )
        from samples.vision.ultralytics_yolo.runtime.python.model_runner import (
            ModelRunner,
        )

        runner, physical, quants, contract = fixture(sdk=True, channels=True)
        for info in quants.values():
            info.zero_point.fill(0)
        metadata = RuntimeMetadata.from_runtime(runner.model)
        binding = bind_model(runner.binding.selection, metadata)
        return ModelRunner(runner.model, binding, metadata), physical, quants, contract

    def test_imoonlab_float_source_uses_the_same_dfl_decoder(self):
        runner, physical, _, contract = fixture()
        outputs = {n: (v.astype(np.float32) - 3) * 0.25 for n, v in physical.items()}
        source = source_task(
            "yolov13_imoonlab", "yolov13", "YoloV13", "YOLOv13Config", {}, outputs
        )
        expected = source.post_process({"m": outputs}, 64, 64)
        task = YoloDetect(
            YoloDetectConfig(
                "fixture.bin", classes_num=1, contract=contract, nms_thres=0.45
            ),
            runner=runner,
        )
        actual = task.post_process(outputs, 64, 64)
        for a, b in zip(actual, expected):
            np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-5)
