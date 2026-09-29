"""Run the pinned original X5/S classification stages without loading an SDK."""

import json
from pathlib import Path
import sys, types, unittest
from unittest.mock import patch
import numpy as np
from source_reference import load_source
from test_classification_binding import fixture


def source_task(target):
    base = Path(__file__).parent / "fixtures"
    facts = json.loads((base / "classification_sources.json").read_text())[target]
    hashes = {p: f["sha256"] for p, f in facts["files"].items()}
    modules = {
        n: types.ModuleType(n)
        for n in [
            "utils",
            "utils.py_utils",
            "utils.py_utils.postprocess",
            "hbm_runtime",
        ]
    }
    modules["utils"].__path__ = []
    modules["utils.py_utils"].__path__ = []
    modules["utils"].py_utils = modules["utils.py_utils"]
    with patch.dict(sys.modules, modules), patch.object(sys, "path", list(sys.path)):
        pre = load_source(
            f"{target}_cls_preprocess.py",
            "utils.py_utils.preprocess",
            base=base,
            hashes=hashes,
        )
        modules["utils.py_utils"].preprocess = pre
        module = load_source(
            f"{target}_cls_source.py", f"fixed_{target}_cls", base=base, hashes=hashes
        )
    typ = module.UltralyticsYOLOCls if target == "x5" else module.YoloCls
    model = typ.__new__(typ)
    model.cfg = types.SimpleNamespace(topk=5, resize_type=0)
    model.input_h = model.input_w = 224
    model.model_name = "m"
    model.output_names = ["opaque"]
    model.input_names = ["image"] if target == "x5" else ["y", "uv"]
    return model


class SourceClassification(unittest.TestCase):
    def test_actual_original_preprocess_and_postprocess(self):
        rng = np.random.default_rng(16)
        for target in ("x5", "s"):
            source = source_task(target)
            model, values, _, _, _ = fixture(
                "x5" if target == "x5" else "s100", shape=(1, 1000, 1, 1)
            )
            for resize in (0, 1):
                source.cfg.resize_type = model.cfg.resize_type = resize
                for shape in ((37, 59, 3), (101, 31, 3)):
                    img = rng.integers(0, 256, shape, dtype=np.uint8)
                    expected = source.pre_process(img)["m"]
                    actual = model.pre_process(img)["m"]
                    for name in expected:
                        np.testing.assert_array_equal(actual[name], expected[name])
            for pattern in ("random", "tie", "extreme"):
                values[:] = rng.normal(size=values.shape) if pattern == "random" else 0
                if pattern == "extreme":
                    values.reshape(-1)[:2] = (-1000, 1000)
                expected = source.post_process({"m": {"opaque": values}})
                self.assertEqual(model.post_process(model.forward({})), expected)


if __name__ == "__main__":
    unittest.main()
