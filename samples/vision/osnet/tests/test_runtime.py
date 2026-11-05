import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from samples.vision.osnet.runtime.python.matching import OSNet, OSNetConfig


class FakeRuntime:
    def __init__(self):
        self.model_names = ["osnet"]
        self.input_names = {"osnet": ["images"]}
        self.output_names = {"osnet": ["embedding"]}
        self.input_shapes = {"osnet": {"images": (1, 256, 128, 3)}}
        self.input_dtypes = {"osnet": {"images": SimpleNamespace(name="RGB")}}
        self.output_shapes = {"osnet": {"embedding": (1, 1, 1, 512)}}
        self.output_dtypes = {"osnet": {"embedding": SimpleNamespace(name="F32")}}
        self.feature = np.ones((1, 1, 1, 512), np.float32)

    def run(self, inputs):
        self.received = inputs
        return {"osnet": {"embedding": self.feature}}


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.runtime = FakeRuntime()
        self.model = OSNet(OSNetConfig(Path("unused.bin")), runtime=self.runtime)

    def test_predict_matches_public_stages(self):
        crop = Image.new("RGB", (101, 75), (31, 77, 211))
        staged = self.model.postprocess(self.model.infer(self.model.preprocess(crop)))
        np.testing.assert_array_equal(self.model.predict(crop), staged)

    def test_rejects_x5_tensor_metadata_for_s_target(self):
        with self.assertRaises(ValueError):
            OSNet(OSNetConfig(Path("unused.bin"), target="s600"), runtime=self.runtime)

    def test_s_targets_use_normalized_f32_nchw(self):
        for target in ("s100", "s100p", "s600"):
            runtime = FakeRuntime()
            runtime.input_shapes["osnet"]["images"] = (1, 3, 256, 128)
            runtime.input_dtypes["osnet"]["images"] = SimpleNamespace(name="F32")
            model = OSNet(OSNetConfig(Path("unused.hbm"), target=target), runtime=runtime)
            tensor = model.preprocess(Image.new("RGB", (64, 128), (255, 128, 0)))["osnet"]["images"]
            self.assertEqual(tensor.dtype, np.float32)
            self.assertEqual(tensor.shape, (1, 3, 256, 128))
            self.assertTrue(tensor.flags.c_contiguous)
            np.testing.assert_allclose(tensor[0, :, 0, 0], [(1-.485)/.229, (128/255-.456)/.224, (0-.406)/.225], atol=1e-6)
            np.testing.assert_allclose(np.linalg.norm(model.predict(Image.new("RGB", (9, 7)))), 1, atol=1e-6)

    def test_rgb_payload_is_signed_planar_despite_nhwc_metadata(self):
        crop = Image.new("RGB", (64, 128), (255, 128, 0))
        tensor = self.model.preprocess(crop)["osnet"]["images"]
        self.assertEqual(tensor.shape, (1, 256, 128, 3))
        self.assertEqual(tensor.dtype, np.int8)
        self.assertTrue(tensor.flags.c_contiguous)
        planes = tensor.reshape(3, 256, 128)
        np.testing.assert_array_equal(planes[0], 127)
        np.testing.assert_array_equal(planes[1], 0)
        np.testing.assert_array_equal(planes[2], -128)

    def test_predict_normalizes_without_mutating_runtime_output(self):
        self.runtime.feature.flat[:2] = [3, 4]
        self.runtime.feature.flat[2:] = 0
        feature = self.model.predict(np.zeros((25, 17, 3), np.uint8))
        np.testing.assert_allclose(feature[:2], [0.6, 0.8])
        self.assertEqual(self.runtime.feature.flat[0], 3)
        self.assertEqual(feature.shape, (512,))

    def test_raw_preserves_feature_values(self):
        model = OSNet(OSNetConfig(Path("unused"), normalize=False), runtime=self.runtime)
        np.testing.assert_array_equal(model(Image.new("RGB", (4, 5))), np.ones(512, np.float32))

    def test_rejects_wrong_input_and_output_metadata(self):
        for attribute, value in [("input_shapes", {"osnet": {"images": (1, 3, 256, 128)}}),
                                 ("input_dtypes", {"osnet": {"images": SimpleNamespace(name="F32")}}),
                                 ("output_shapes", {"osnet": {"embedding": (1, 1000)}}),
                                 ("output_dtypes", {"osnet": {"embedding": SimpleNamespace(name="S32")}})]:
            with self.subTest(attribute=attribute):
                runtime = FakeRuntime()
                setattr(runtime, attribute, value)
                with self.assertRaises(ValueError):
                    OSNet(OSNetConfig(Path("unused")), runtime=runtime)

    def test_rejects_nonfinite_or_integer_features(self):
        for values in [np.full(512, np.nan, np.float32), np.zeros(512, np.int32), np.zeros(256, np.float32)]:
            with self.subTest(dtype=values.dtype, size=values.size), self.assertRaises(ValueError):
                self.model.postprocess({"osnet": {"embedding": values}})

    def test_rejects_invalid_image_arrays(self):
        for image in [np.zeros((10, 10, 3), np.float32), np.zeros((10, 10), np.uint8), np.zeros((0, 1, 3), np.uint8)]:
            with self.assertRaises(ValueError):
                self.model.preprocess(image)


if __name__ == "__main__":
    unittest.main()
