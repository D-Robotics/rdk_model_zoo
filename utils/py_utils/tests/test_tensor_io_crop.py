"""Shorter-edge resize + center crop (resize_type 2) in the shared preprocessing.

The geometry must equal the host evaluation path exactly, because the published
models were calibrated and evaluated with it and the runtime has to feed the same
bytes. Expected sizes and offsets below are worked out by hand from the timm rule:
resize so the shorter edge is `shorter` (the longer edge is truncated with `int`),
then crop at `round((resized - size) / 2)`.
"""

from __future__ import annotations

from pathlib import Path
import builtins
import unittest
from unittest import mock

import numpy as np

try:
    from PIL import Image
except ImportError:  # pragma: no cover - the host environments used for CI have Pillow
    Image = None

from utils.py_utils import cls_binding, tensor_io
from utils.py_utils.model_runner import RuntimeModelRunner


def _image(width: int, height: int, seed: int = 0) -> np.ndarray:
    """Return a deterministic noisy BGR image so resampling errors are visible."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)


@unittest.skipIf(Image is None, "Pillow is required for resize_type 2")
class ShorterEdgeCenterCropTests(unittest.TestCase):
    def test_hand_computed_geometry(self):
        # (width, height, shorter, expected resized (w, h), expected (left, top))
        cases = (
            (300, 200, 256, (384, 256), (80, 16)),    # landscape: 256*300/200 = 384
            (200, 300, 256, (256, 384), (16, 80)),    # portrait
            (500, 500, 256, (256, 256), (16, 16)),    # square
            (10, 20, 256, (256, 512), (16, 144)),     # upscaling a tiny image
        )
        for width, height, shorter, resized, offsets in cases:
            with self.subTest(width=width, height=height, shorter=shorter):
                image = Image.fromarray(np.zeros((height, width, 3), dtype=np.uint8))
                crop, got_resized, got_offsets = tensor_io.resize_shorter_center_crop(
                    image, 224, shorter)
                self.assertEqual(crop.size, (224, 224))
                self.assertEqual(got_resized, resized)
                self.assertEqual(got_offsets, offsets)

    def test_offsets_round_half_to_even_like_python_round(self):
        # 235 -> (235 - 224) / 2 = 5.5 rounds to 6 under round-half-to-even
        # (Python's round), the same rule the host evaluation uses.
        image = Image.fromarray(np.zeros((200, 300, 3), dtype=np.uint8))
        _, resized, offsets = tensor_io.resize_shorter_center_crop(image, 224, 235)
        self.assertEqual(resized, (352, 235))
        self.assertEqual(offsets, (64, 6))

    def test_matches_independent_pillow_reference(self):
        for width, height in ((640, 480), (480, 640), (333, 333), (1000, 300)):
            with self.subTest(width=width, height=height):
                rgb = Image.fromarray(_image(width, height, seed=width)[:, :, ::-1])
                crop, resized, (left, top) = tensor_io.resize_shorter_center_crop(rgb, 224, 256)
                if width <= height:
                    expected_size = (256, int(256 * height / width))
                else:
                    expected_size = (int(256 * width / height), 256)
                reference = rgb.resize(expected_size, Image.BICUBIC).crop(
                    (left, top, left + 224, top + 224))
                self.assertEqual(resized, expected_size)
                np.testing.assert_array_equal(np.asarray(crop), np.asarray(reference))

    def test_rejects_shorter_edge_below_crop_size(self):
        image = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))
        with self.assertRaises(ValueError):
            tensor_io.resize_shorter_center_crop(image, 224, 223)
        with self.assertRaises(ValueError):
            tensor_io.resize_shorter_center_crop(image, 0, 256)


@unittest.skipIf(Image is None, "Pillow is required for resize_type 2")
class ResizeBgrPolicyTests(unittest.TestCase):
    def test_runtime_geometry_equals_host_evaluation_geometry(self):
        from utils.py_utils.classification_host import prepare_rgb

        contract = {
            "size": 224, "crop_pct": 0.875, "geometry": "resize_shorter_center_crop",
            "interpolation": "pil_bicubic", "color": "RGB", "layout": "NCHW",
            "dtype": "float32", "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225],
            "batch": 1, "class_count": 1000, "output": "logits",
        }
        for width, height in ((640, 480), (480, 640), (320, 320)):
            with self.subTest(width=width, height=height):
                bgr = _image(width, height, seed=height)
                host_rgb = prepare_rgb(Image.fromarray(bgr[:, :, ::-1]), contract)
                runtime_bgr, _ = tensor_io.resize_bgr(
                    bgr, 224, 224, resize_type=2, resize_shorter=256)
                np.testing.assert_array_equal(runtime_bgr, host_rgb[:, :, ::-1])
                self.assertEqual(runtime_bgr.dtype, np.uint8)
                self.assertTrue(runtime_bgr.flags["C_CONTIGUOUS"])

    def test_transform_records_resize_and_crop_without_padding(self):
        bgr = _image(300, 200)
        _, transform = tensor_io.resize_bgr(bgr, 224, 224, resize_type=2, resize_shorter=256)
        self.assertEqual((transform.original_height, transform.original_width), (200, 300))
        self.assertEqual((transform.resized_height, transform.resized_width), (256, 384))
        self.assertEqual((transform.crop_x, transform.crop_y), (80.0, 16.0))
        self.assertEqual(
            (transform.pad_top, transform.pad_bottom, transform.pad_left, transform.pad_right),
            (0, 0, 0, 0))
        self.assertAlmostEqual(transform.scale_x, 384 / 300)
        self.assertAlmostEqual(transform.scale_y, 256 / 200)

    def test_input_array_is_not_modified(self):
        bgr = _image(320, 240)
        before = bgr.copy()
        tensor_io.resize_bgr(bgr, 224, 224, resize_type=2, resize_shorter=256)
        np.testing.assert_array_equal(bgr, before)

    def test_invalid_arguments(self):
        bgr = _image(320, 240)
        with self.assertRaisesRegex(ValueError, "square"):
            tensor_io.resize_bgr(bgr, 224, 160, resize_type=2, resize_shorter=256)
        with self.assertRaisesRegex(ValueError, "resize_shorter"):
            tensor_io.resize_bgr(bgr, 224, 224, resize_type=2, resize_shorter=0)
        with self.assertRaisesRegex(ValueError, "resize_shorter"):
            tensor_io.resize_bgr(bgr, 224, 224, resize_type=2, resize_shorter=200)
        with self.assertRaisesRegex(ValueError, "expected 0, 1 or 2"):
            tensor_io.resize_bgr(bgr, 224, 224, resize_type=3)

    def test_other_policies_do_not_need_resize_shorter(self):
        bgr = _image(320, 240)
        stretched, _ = tensor_io.resize_bgr(bgr, 224, 224, resize_type=0)
        letterboxed, _ = tensor_io.resize_bgr(bgr, 224, 224, resize_type=1)
        self.assertEqual(stretched.shape, (224, 224, 3))
        self.assertEqual(letterboxed.shape, (224, 224, 3))


class MissingPillowTests(unittest.TestCase):
    def test_policy_two_names_the_missing_dependency(self):
        real_import = builtins.__import__

        def blocked(name, *args, **kwargs):
            if name == "PIL" or name.startswith("PIL."):
                raise ImportError("No module named 'PIL'")
            return real_import(name, *args, **kwargs)

        with mock.patch.object(builtins, "__import__", blocked):
            with self.assertRaisesRegex(RuntimeError, "Pillow is required"):
                tensor_io.resize_bgr(_image(64, 64), 32, 32, resize_type=2, resize_shorter=40)

    def test_policies_zero_and_one_never_import_pillow(self):
        real_import = builtins.__import__

        def blocked(name, *args, **kwargs):
            if name == "PIL" or name.startswith("PIL."):
                raise AssertionError("Pillow must not be imported for resize types 0 and 1")
            return real_import(name, *args, **kwargs)

        with mock.patch.object(builtins, "__import__", blocked):
            for resize_type in (0, 1):
                tensor_io.resize_bgr(_image(64, 48), 32, 32, resize_type=resize_type)


class ContractValidationTests(unittest.TestCase):
    def _record(self):
        return cls_binding.AssetRecord(
            asset_id="x5:demo:demo.bin", variant="demo", target="x5", filename="demo.bin",
            model_format="bin", source_manifest="docs/release/x5/models.yaml", sample_id="demo")

    def _table(self, facts):
        return cls_binding.SampleBindingTable(
            sample_dir=Path("."), manifest_rows=(), filename_variants={},
            default_variant="demo", facts={("demo", "x5"): facts})

    def test_contract_carries_resize_shorter(self):
        facts = cls_binding.VariantFacts(
            input_height=224, input_width=224, resize_type=2, resize_shorter=256)
        contract = cls_binding.contract_for(self._table(facts), self._record())
        self.assertEqual((contract.resize_type, contract.resize_shorter), (2, 256))

    def test_resize_shorter_defaults_to_unused(self):
        facts = cls_binding.VariantFacts(input_height=224, input_width=224)
        contract = cls_binding.contract_for(self._table(facts), self._record())
        self.assertEqual((contract.resize_type, contract.resize_shorter), (1, 0))

    def test_inconsistent_rows_fail_when_the_contract_is_built(self):
        for facts in (
            cls_binding.VariantFacts(input_height=224, input_width=224, resize_type=2),
            cls_binding.VariantFacts(
                input_height=224, input_width=224, resize_type=2, resize_shorter=200),
            cls_binding.VariantFacts(input_height=224, input_width=224, resize_type=7),
        ):
            with self.subTest(facts=facts):
                with self.assertRaises(cls_binding.ManifestAssetError):
                    cls_binding.contract_for(self._table(facts), self._record())


class LocalRunnerTests(unittest.TestCase):
    def _runner(self, **overrides):
        arguments = dict(
            target="x5", input_size=(224, 224), class_count=1000, resize_type=2,
            resize_shorter=256, runtime=object())
        arguments.update(overrides)
        return RuntimeModelRunner.from_file("/nonexistent/model.bin", **arguments)

    def test_policy_two_is_accepted_and_recorded(self):
        contract = self._runner().selection.contract
        self.assertEqual((contract.resize_type, contract.resize_shorter), (2, 256))

    def test_policy_two_requires_a_square_input_and_a_large_enough_edge(self):
        with self.assertRaises(ValueError):
            self._runner(input_size=(224, 160))
        with self.assertRaises(ValueError):
            self._runner(resize_shorter=0)
        with self.assertRaises(ValueError):
            self._runner(resize_shorter=223)

    def test_other_policies_ignore_resize_shorter(self):
        contract = self._runner(resize_type=1, resize_shorter=0).selection.contract
        self.assertEqual((contract.resize_type, contract.resize_shorter), (1, 0))


if __name__ == "__main__":
    unittest.main()
