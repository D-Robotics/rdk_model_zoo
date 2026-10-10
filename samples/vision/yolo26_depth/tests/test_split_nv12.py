"""Physical split NV12 contract used by the S full depth artifacts."""
from types import SimpleNamespace
import unittest
import numpy as np
from utils.py_utils.image import bgr_to_nv12_planes
from samples.vision.yolo26_depth.runtime.python.cli import resolve_selection
from samples.vision.yolo26_depth.runtime.python.yolo26_depth import bind_model
from samples.vision.yolo26_depth.runtime.python.yolo26_depth import RuntimeModelRunner
from samples.vision.yolo26_depth.runtime.python.yolo26_depth import Yolo26DepthTask


def metadata():
    return dict(model_name="depth", model_names=["depth"], input_names=["images_uv", "images_y"],
                input_shapes={"images_y": (1,768,768,1), "images_uv": (1,384,384,2)},
                input_dtypes={"images_y": "uint8", "images_uv": "uint8"}, output_names=["output0"],
                output_shapes={"output0": (1,192,192,1)}, output_dtypes={"output0": "float32"})


class SplitNV12Tests(unittest.TestCase):
    def test_all_nine_full_s_profiles_and_plane_bytes(self):
        image = np.arange(768*768*3, dtype=np.uint8).reshape(768,768,3)
        y, uv = bgr_to_nv12_planes(image)
        for target in ("s100", "s100p", "s600"):
            for variant in ("n", "s", "m"):
                selected = resolve_selection(target, variant=variant)
                bound = bind_model(selected, metadata())
                runtime = SimpleNamespace(**metadata())
                runtime.run = lambda inputs: {"depth": {"output0": np.zeros((1,192,192,1), np.float32)}}
                runner = RuntimeModelRunner(selected, runtime=runtime)
                task = Yolo26DepthTask(runner=runner, binding=bound)
                prepared = task.preprocess(image)
                np.testing.assert_array_equal(prepared.tensors["images_y"].reshape(y.shape), y)
                np.testing.assert_array_equal(prepared.tensors["images_uv"].reshape(uv.shape), uv)
                result = task.predict(image)
                np.testing.assert_allclose(result.depth_native, 1)
                invalid = dict(prepared.tensors)
                invalid["images_uv"] = invalid["images_uv"].astype(np.float32)
                with self.assertRaises(ValueError):
                    task.infer(invalid)

    def test_wrong_plane_contract_is_rejected(self):
        selected = resolve_selection("s100", variant="n")
        for changes in ({"input_shapes": {"images_y": (1,768,768,1), "images_uv": (1,384,768,1)}},
                        {"input_dtypes": {"images_y": "uint8", "images_uv": "nv12"}},
                        {"input_names": ["images_y", "images_uv", "extra"]}):
            with self.assertRaises(ValueError):
                bind_model(selected, {**metadata(), **changes})
        for target, variant in (("x5", "n"), ("s100", "l")):
            with self.assertRaises(ValueError):
                bind_model(resolve_selection(target, variant=variant), metadata())
