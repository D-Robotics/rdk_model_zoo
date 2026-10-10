"""Compare the real previous decoder; SDK loading is not part of this fixture."""

from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch
import hashlib
import importlib.util
import utils.py_utils as common_utils
import utils.py_utils.preprocess as common_preprocess
import utils.py_utils.postprocess as common_postprocess
import json
import sys
import unittest
import numpy as np
from test_obb_binding import fixture

FIXTURE = Path(__file__).parent / "fixtures/unified_yolo26_obb_before_runner.py"


def previous(model):
    metadata = json.loads(FIXTURE.with_suffix(".json").read_text())
    if hashlib.sha256(FIXTURE.read_bytes()).hexdigest() != metadata["sha256"]:
        raise AssertionError("OBB reference bytes changed")
    stub = ModuleType("yolo26_common")
    stub.Yolo26Runtime = type(
        "Yolo26Runtime",
        (),
        {"logit_threshold": staticmethod(lambda score: -np.log(1 / score - 1))},
    )
    spec = importlib.util.spec_from_file_location("obb_previous_fixture", FIXTURE)
    module = importlib.util.module_from_spec(spec)
    runtime_path = str(FIXTURE.parents[2] / "runtime/python")
    # The frozen fixture keeps its historical flat imports; the consolidated
    # package modules are exposed under those names for its execution only.
    from samples.vision.ultralytics_yolo.runtime.python import cli
    with patch.object(sys, "path", [runtime_path, *sys.path]), patch.dict(
        sys.modules, {"yolo26_common": stub, spec.name: module,
                      "yolo_platform": cli,
                      "rdk_yolo_utils": common_utils,
                      "rdk_yolo_utils.preprocess": common_preprocess,
                      "rdk_yolo_utils.postprocess": common_postprocess}
    ):
        spec.loader.exec_module(module)
    old = object.__new__(module.YOLO26OBB)
    old.cfg = SimpleNamespace(**vars(model.cfg))
    old.cfg.platform = model.profile
    old.model_name = "m"
    old.input_w = old.input_h = 64
    old.output_names = [
        f"{kind}_{s}" for s in (8, 16, 32) for kind in ("cls", "box", "angle")
    ]
    old.map_idx = {s: (3 * i, 3 * i + 1, 3 * i + 2) for i, s in enumerate((8, 16, 32))}
    old.grids = {
        s: np.stack(np.indices((64 // s, 64 // s))[::-1], axis=-1)
        .reshape(-1, 2)
        .astype(np.float32)
        + 0.5
        for s in (8, 16, 32)
    }
    old.angle_offset_rad = np.deg2rad(model.cfg.angle_offset)
    return old


class OBBReference(unittest.TestCase):
    def test_previous_decoder_platform_policies_and_geometry(self):
        # Exact-resize cases distinguish preserved arithmetic from the separately
        # tested intentional integer-letterbox correction.
        for target in ("x5", "s100"):
            for resize in (0, 1):
                for regularize in (False, True):
                    with self.subTest(
                        target=target, resize=resize, regularize=regularize
                    ):
                        model, raw, _, _ = fixture(target)
                        model.cfg.resize_type = resize
                        model.cfg.regularize = regularize
                        model.cfg.angle_sign = -1
                        model.cfg.angle_offset = 20
                        raw["cls_8"][0, 3, 4, 3] = 5
                        raw["box_8"][0, 3, 4] = [-1, 3, -1, 3]
                        raw["angle_8"][0, 3, 4, 0] = 0.7
                        raw["cls_16"][0, 1, 1, 2] = 4
                        raw["box_16"][0, 1, 1] = [1, 2, 3, 4]
                        a = previous(model).post_process({"m": raw}, 64, 32)
                        b = model.post_process(model.forward({}), 64, 32)
                        self.assertEqual([v["id"] for v in a], [v["id"] for v in b])
                        np.testing.assert_allclose(
                            [v["score"] for v in a], [v["score"] for v in b], rtol=1e-6
                        )
                        np.testing.assert_allclose(
                            [v["rrect"] for v in a],
                            [v["rrect"] for v in b],
                            rtol=1e-6,
                            atol=1e-5,
                        )

    def test_invalid_angle_controls_and_target_conflict(self):
        from samples.vision.ultralytics_yolo.runtime.python.obb import (
            YOLO26OBB,
            YOLO26OBBConfig,
        )

        model, _, _, _ = fixture()
        with self.assertRaisesRegex(ValueError, "target conflicts"):
            YOLO26OBB(YOLO26OBBConfig("stub", platform="x5"), runner=model.runner)
        for attr in ("angle_sign", "angle_offset"):
            with self.subTest(attr=attr):
                old = getattr(model.cfg, attr)
                setattr(model.cfg, attr, np.inf)
                with self.assertRaisesRegex(ValueError, "Angle controls"):
                    model.predict(np.zeros((64, 64, 3), np.uint8))
                setattr(model.cfg, attr, old)


if __name__ == "__main__":
    unittest.main()
