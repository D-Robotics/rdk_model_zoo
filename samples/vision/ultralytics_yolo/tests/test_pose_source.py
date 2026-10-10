"""Execute fixed S/X5 source code and verify their different public score domains."""

import json
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch
import numpy as np
from test_pose_binding import fixture
import test_pose_binding as checks
from source_reference import load_source, ROOT
from samples.vision.ultralytics_yolo.runtime.python.backend import (
    RuntimeMetadata,
    bind_model,
)
from samples.vision.ultralytics_yolo.runtime.python.backend import ModelRunner
from samples.vision.ultralytics_yolo.runtime.python.pose import YoloPose


def x5_source_task(names):
    base = Path(__file__).parent / "fixtures"
    provenance = json.loads((base / "x5_pose_source.json").read_text())
    packages = {
        name: types.ModuleType(name)
        for name in ["utils", "utils.py_utils", "hbm_runtime"]
    }
    packages["utils"].__path__ = []
    packages["utils.py_utils"].__path__ = []
    packages["utils"].py_utils = packages["utils.py_utils"]
    packages["hbm_runtime"].QuantParams = object
    with patch.dict(sys.modules, packages), patch.object(sys, "path", list(sys.path)):
        for helper in ("nn_math", "preprocess", "postprocess"):
            module = load_source(
                f"utils/py_utils/{helper}.py",
                f"utils.py_utils.{helper}",
                base=ROOT / "platforms/x5",
                hashes=provenance["helpers"],
            )
            setattr(packages["utils.py_utils"], helper, module)
        module = load_source(
            "x5_pose_source.py",
            "fixed_x5_pose",
            base=base,
            hashes={"x5_pose_source.py": provenance["sha256"]},
        )
    source = module.UltralyticsYOLOPose.__new__(module.UltralyticsYOLOPose)
    source.cfg = module.UltralyticsYOLOPoseConfig("fixture.bin")
    source.model_name = "m"
    source.output_names = list(names)
    source.input_h = source.input_w = 64
    source.conf_thres_raw = -np.log(1 / source.cfg.score_thres - 1)
    source.weights_static = np.arange(16, dtype=np.float32)[None, None, :]
    return source


class PoseSource(unittest.TestCase):
    def test_x5_source_three_tuple_probability_domain(self):
        task, raw, _ = fixture(target="x5", quantized=False)
        source = x5_source_task(raw)
        for resize in (0, 1):
            source.cfg.resize_type = task.cfg.resize_type = resize
            for width, height in ((64, 64), (128, 64)):
                expected = source.post_process({"m": raw}, width, height)
                result = task.post_process(task.forward({}), width, height)
                actual = (
                    result[0],
                    result[1],
                    np.concatenate([result[3], result[4]], axis=-1),
                )
                for a, b in zip(actual, expected):
                    np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-5)
