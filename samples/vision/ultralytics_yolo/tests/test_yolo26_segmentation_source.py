"""Pinned mask source and actual compatibility adapter checks."""

import sys, importlib.util, unittest, json, types
from unittest.mock import patch
from pathlib import Path
import numpy as np
from source_reference import ROOT, load_source
from test_yolo26_segmentation_binding import fixture


def source_task(target, size=64):
    base = Path(__file__).parent / "fixtures"
    facts = json.loads((base / "yolo26_seg_sources.json").read_text())[target]
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
        filename = f"{target}_yolo26_seg_source.py"
        mod = load_source(
            filename,
            f"fixed_{target}_yolo26_seg",
            base=base,
            hashes={filename: facts["sha256"]},
        )
    model = mod.YOLO26Seg.__new__(mod.YOLO26Seg)
    model.cfg = types.SimpleNamespace(
        strides=[8, 16, 32],
        resize_type=1,
        score_thres=0.25,
        nms_thres=0.65,
        classes_num=1,
    )
    model.input_w = model.input_h = size
    model.model_name = "m"
    model.output_names = [
        f"{kind}_{stride}" for stride in (8, 16, 32) for kind in ("cls", "box", "mces")
    ] + ["protos"]
    return model


class SegmentationSource(unittest.TestCase):
    def test_original_s_mask_algorithm_and_roi_are_preserved(self):
        source = source_task("s")
        model, raw, _, _ = fixture()
        for mode in (0, 1):
            model.cfg.resize_type = source.cfg.resize_type = mode
            for w, h in ((64, 64), (128, 64)):
                expected = source.post_process({"m": raw}, w, h)
                actual = model.post_process(model.forward({}), w, h)
                for a, b in zip(expected[:3], actual[:3]):
                    np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-5)
                for a, b in zip(expected[3], actual[3]):
                    np.testing.assert_array_equal(a, b)

    def test_old_x5_full_mask_misses_letterbox_removal(self):
        source = source_task("x5", 640)
        model, raw, _, _ = fixture("x5", size=640)
        raw["cls_8"].fill(-20)
        raw["cls_8"][0, 40, 40, 0] = 6
        raw["protos"].fill(1)
        old = source.post_process({"m": raw}, 640, 320)
        new = model.post_process(model.forward({}), 640, 320)
        np.testing.assert_allclose(old[0], new[0], atol=1e-5)
        current = np.zeros((320, 640), bool)
        x1, y1, x2, y2 = new[0][0].astype(int)
        current[y1:y2, x1:x2] = new[3][0]
        self.assertFalse(np.array_equal(old[3][0], current))
        self.assertEqual(int(current.sum()), 256)

    def test_actual_x5_adapter_supports_explicit_geometry(self):
        path = (
            ROOT
            / "platforms/x5/samples/vision/ultralytics_yolo26/runtime/python/yolo26_seg.py"
        )
        spec = importlib.util.spec_from_file_location("legacy_y26_seg_test", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        unified, _, _, _ = fixture("x5")
        legacy = module.YOLO26Seg(
            module.YOLO26SegConfig("stub", classes_num=1), runner=unified.runner
        )
        image = np.zeros((37, 59, 3), np.uint8)
        boxes, scores, ids, full = legacy.predict(image)
        expected = unified.predict(image)
        for a, b in zip((boxes, scores, ids), expected[:3]):
            np.testing.assert_array_equal(a, b)
        self.assertEqual(full.shape, (1, 37, 59))
        self.assertEqual(full.dtype, np.bool_)
        x1, y1, x2, y2 = boxes[0].astype(int)
        np.testing.assert_array_equal(full[0, y1:y2, x1:x2], expected[3][0])


if __name__ == "__main__":
    unittest.main()
