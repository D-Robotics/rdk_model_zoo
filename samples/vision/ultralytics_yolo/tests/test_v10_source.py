"""Execute the original pinned S v10 decoder; distinguish geometry corrections."""

import json, sys, types, unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from source_reference import load_source, ROOT
from test_v10_stages import fixture


def source_task():
    base = Path(__file__).parent / "fixtures"
    facts = json.loads((base / "s_v10_source.json").read_text())
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
                base=ROOT / "platforms/s",
                hashes=facts["helpers"],
            )
            setattr(packages["utils.py_utils"], helper, mod)
        module = load_source(
            "s_v10_source.py",
            "fixed_s_v10",
            base=base,
            hashes={"s_v10_source.py": facts["sha256"]},
        )
    model = module.YoloV10Detect.__new__(module.YoloV10Detect)
    model.cfg = types.SimpleNamespace(
        strides=[8, 16, 32], resize_type=1, score_thres=0.25, anchor_sizes=[8, 4, 2]
    )
    model.input_w = model.input_h = 64
    model.model_name = "m"
    model.input_names = ["y", "uv"]
    model.output_names = [
        f"{kind}_{stride}" for stride in (8, 16, 32) for kind in ("cls", "box")
    ]
    model.anchor_sizes = [8, 4, 2]
    model.weights_static = np.arange(16, dtype=np.float32)[None, None, :]
    return model


class SourceV10(unittest.TestCase):
    def test_source_preprocessing_and_nonrounded_postprocessing(self):
        source = source_task()
        model, raw = fixture()
        rng = np.random.default_rng(5)
        for mode in (0, 1):
            source.cfg.resize_type = model.cfg.resize_type = mode
            for shape in ((37, 59, 3), (64, 128, 3)):
                img = rng.integers(0, 256, size=shape, dtype=np.uint8)
                before = source.pre_process(img)["m"]
                after = model.pre_process(img)["m"]
                for name in before:
                    np.testing.assert_array_equal(before[name], after[name])
            for w, h in ((64, 64), (128, 64)):
                before = source.post_process({"m": raw}, w, h)
                after = model.post_process(model.forward({}), w, h)
                for a, b in zip(before, after):
                    np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-5)
        # Integer resize/padding fixes the old ideal-scale letterbox inverse.
        model.cfg.resize_type = source.cfg.resize_type = 1
        old = source.post_process({"m": raw}, 59, 37)[0]
        new = model.post_process(model.forward({}), 59, 37).boxes
        self.assertFalse(np.allclose(old, new, atol=1e-4))


if __name__ == "__main__":
    unittest.main()
