"""Run pinned X5/S YOLO26 pose sources and the maintained legacy adapter."""

import json, sys, types, unittest, importlib.util
from pathlib import Path
from unittest.mock import patch
import numpy as np
from source_reference import load_source, ROOT
from test_yolo26_pose_binding import fixture


def source_task(target):
    base = Path(__file__).parent / "fixtures"
    facts = json.loads((base / "yolo26_pose_sources.json").read_text())[target]
    packages = {
        n: types.ModuleType(n) for n in ("utils", "utils.py_utils", "hbm_runtime")
    }
    packages["utils"].__path__ = []
    packages["utils.py_utils"].__path__ = []
    packages["utils"].py_utils = packages["utils.py_utils"]
    packages["hbm_runtime"].QuantParams = object
    with patch.dict(sys.modules, packages), patch.object(sys, "path", list(sys.path)):
        for helper in ("nn_math", "preprocess", "postprocess"):
            mod = load_source(
                f"utils/py_utils/{helper}.py",
                f"utils.py_utils.{helper}",
                base=ROOT / "platforms" / target,
                hashes=facts["helpers"],
            )
            setattr(packages["utils.py_utils"], helper, mod)
        filename = f"{target}_yolo26_pose_source.py"
        mod = load_source(
            filename,
            f"fixed_{target}_yolo26_pose",
            base=base,
            hashes={filename: facts["sha256"]},
        )
    model = mod.YOLO26Pose.__new__(mod.YOLO26Pose)
    model.cfg = types.SimpleNamespace(
        strides=[8, 16, 32], resize_type=1, score_thres=0.25, nms_thres=0.65
    )
    model.input_w = model.input_h = 64
    model.model_name = "m"
    model.conf_raw = -np.log(3)
    model.output_names = [
        f"{kind}_{stride}" for stride in (8, 16, 32) for kind in ("cls", "box", "kpts")
    ]
    model.grids = {
        stride: np.stack(np.indices((64 // stride, 64 // stride))[::-1], axis=-1)
        .reshape(-1, 2)
        .astype(np.float32)
        + 0.5
        for stride in (8, 16, 32)
    }
    return model


class SourceDirectPose(unittest.TestCase):
    def test_both_original_decoders_unrounded_geometry(self):
        for target in ("x5", "s"):
            source = source_task(target)
            model, raw, _, _ = fixture()
            for mode in (0, 1):
                model.cfg.resize_type = source.cfg.resize_type = mode
                for w, h in ((64, 64), (128, 64)):
                    actual = model.post_process(model.forward({}), w, h)
                    expected = source.post_process({"m": raw}, w, h)
                    if target == "x5":
                        np.testing.assert_array_equal(
                            actual[0].astype(int),
                            np.stack([r["box"] for r in expected]),
                        )
                        np.testing.assert_allclose(
                            actual[1], [r["score"] for r in expected], rtol=1e-6
                        )
                        kpts = np.stack([r["kpts"] for r in expected])
                    else:
                        for a, b in zip(actual[:3], expected[:3]):
                            np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-5)
                        kpts = expected[3]
                    np.testing.assert_allclose(
                        actual[3], kpts[..., :2], rtol=1e-6, atol=1e-5
                    )
                    np.testing.assert_allclose(
                        actual[4], kpts[..., 2:3], rtol=1e-6, atol=1e-6
                    )

    def test_small_threshold_no_longer_silently_clamped(self):
        source = source_task("s")
        model, raw, _, _ = fixture()
        raw["cls_8"][0, 3, 3, 0] = -14.2
        old = source.post_process({"m": raw}, 64, 64, score_thres=5e-7)
        current = model.post_process(model.forward({}), 64, 64, score_thres=5e-7)
        self.assertEqual(len(old[0]), 0)
        self.assertEqual(len(current[0]), 1)
        self.assertGreater(current[1][0], 5e-7)

    def test_actual_s_legacy_adapter_returns_four_tuple(self):
        path = (
            ROOT
            / "platforms/s/samples/vision/ultralytics_yolo26/runtime/python/yolo26_pose.py"
        )
        spec = importlib.util.spec_from_file_location("legacy_s_y26_pose_test", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        from samples.vision.ultralytics_yolo.runtime.python.yolo_platform import (
            resolve_platform,
        )

        unified, _, _, _ = fixture("s100")
        legacy = module.YOLO26Pose(
            module.YOLO26PoseConfig("stub", platform=resolve_platform("s100")),
            runner=unified.runner,
        )
        image = np.zeros((64, 64, 3), np.uint8)
        actual = legacy.predict(image)
        expected = unified.predict(image)
        self.assertEqual(len(actual), 4)
        for left, right in zip(actual[:3], expected[:3]):
            np.testing.assert_array_equal(left, right)
        np.testing.assert_array_equal(
            actual[3], np.concatenate([expected[3], expected[4]], axis=-1)
        )

    def test_actual_x5_legacy_adapter_returns_list(self):
        path = (
            ROOT
            / "platforms/x5/samples/vision/ultralytics_yolo26/runtime/python/yolo26_pose.py"
        )
        spec = importlib.util.spec_from_file_location("legacy_y26_pose_test", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        unified, _, _, _ = fixture("x5")
        legacy = module.YOLO26Pose(
            module.YOLO26PoseConfig("stub", nms_thres=0.65), runner=unified.runner
        )
        image = np.zeros((64, 64, 3), np.uint8)
        actual = legacy.predict(image)
        expected = unified.predict(image)
        self.assertEqual(len(actual), 1)
        np.testing.assert_array_equal(actual[0]["box"], expected[0][0].astype(int))
        np.testing.assert_allclose(
            actual[0]["kpts"], np.concatenate([expected[3][0], expected[4][0]], axis=-1)
        )


if __name__ == "__main__":
    unittest.main()
