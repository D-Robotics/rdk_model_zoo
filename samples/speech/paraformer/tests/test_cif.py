"""CPU CIF boundary and pinned source parity checks; no board inference."""

import importlib.util
from pathlib import Path
import unittest

import numpy as np
from samples._shared.tests.legacy_platforms import legacy_path, legacy_tree  # noqa: E402

ROOT = Path(__file__).resolve().parents[4]
SOURCE = legacy_path("s/samples/speech/paraformer/conversion/cif_numpy.py")
TARGET = ROOT / "samples/speech/paraformer/runtime/python/cif.py"


def load_function(path):
    spec = importlib.util.spec_from_file_location("cif_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.cif_numpy


class CifTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(TARGET.is_file(), "Unified CPU CIF implementation is missing")
        self.cif = load_function(TARGET)
        self.source = load_function(SOURCE)
        self.alphas = np.zeros((1, 401), dtype=np.float32)
        self.hidden = np.zeros((1, 401, 512), dtype=np.float32)

    def test_empty_fire_returns_fixed_zero_decoder_input(self):
        for total in (0.0, 0.5):
            with self.subTest(total=total):
                self.alphas[0, 0] = total
                frames, count = self.cif(self.alphas, self.hidden, real_T=400)
                self.assertEqual(frames.shape, (1, 100, 512))
                self.assertEqual(frames.dtype, np.float32)
                self.assertEqual(count.dtype, np.int32)
                np.testing.assert_array_equal(count, [0])
                self.assertFalse(frames.any())

    def test_fractional_boundary_conserves_weighted_hidden(self):
        self.alphas[0, :3] = [0.75, 0.75, 0.5]
        self.hidden[0, :3] = np.array([2, 6, 10])[:, None]
        frames, count = self.cif(self.alphas, self.hidden, real_T=3)
        np.testing.assert_array_equal(count, [2])
        np.testing.assert_array_equal(frames[0, 0], np.full(512, 3.0))
        np.testing.assert_array_equal(frames[0, 1], np.full(512, 8.0))
        self.assertFalse(frames[0, 2:].any())

    def test_padding_mask_and_explicit_calibration_mode(self):
        self.alphas[0, 10] = 1
        self.hidden[0, 10] = 5
        masked, count = self.cif(self.alphas, self.hidden, real_T=10)
        self.assertFalse(masked.any())
        np.testing.assert_array_equal(count, [0])
        unmasked, count = self.cif(self.alphas, self.hidden, real_T=None)
        np.testing.assert_array_equal(count, [1])
        np.testing.assert_array_equal(unmasked[0, 0], np.full(512, 5))

    def test_more_than_100_fires_preserves_first_100(self):
        self.alphas.fill(1)
        self.hidden[:] = np.arange(401)[None, :, None]
        frames, count = self.cif(self.alphas, self.hidden, real_T=400)
        np.testing.assert_array_equal(count, [100])
        np.testing.assert_array_equal(frames[0, :, 0], np.arange(100))

    def test_source_parity_without_mutation_or_output_alias(self):
        for seed in range(6):
            rng = np.random.default_rng(seed)
            a = rng.uniform(0, 0.8, (1, 401)).astype(np.float32)
            h = rng.normal(size=(1, 401, 512)).astype(np.float32)
            before_a, before_h = a.copy(), h.copy()
            for length in (1, 30, 400, None):
                if length == 1:
                    a[0, 0] = 1
                    before_a = a.copy()
                actual = self.cif(a, h, real_T=length)
                expected = self.source(a, h, real_T=length)
                for got, want in zip(actual, expected):
                    np.testing.assert_array_equal(got, want)
                    self.assertEqual(got.tobytes(), want.tobytes())
                self.assertFalse(np.shares_memory(actual[0], h))
                np.testing.assert_array_equal(a, before_a)
                np.testing.assert_array_equal(h, before_h)

    def test_inference_length_must_be_explicit_and_valid(self):
        with self.assertRaises(TypeError):
            self.cif(self.alphas, self.hidden)
        for length in (-1, 401, 1.5, True, "10"):
            with self.subTest(length=length), self.assertRaises(
                (TypeError, ValueError)
            ):
                self.cif(self.alphas, self.hidden, real_T=length)
        _, count = self.cif(self.alphas, self.hidden, real_T=np.int64(0))
        np.testing.assert_array_equal(count, [0])

    def test_invalid_tensor_contract_is_rejected(self):
        cases = [
            (self.alphas.astype(np.float64), self.hidden),
            (self.alphas, self.hidden.astype(np.int16)),
            (np.zeros((2, 401), np.float32), np.zeros((2, 401, 512), np.float32)),
            (self.alphas[:, :-1], self.hidden),
            (self.alphas, self.hidden[:, :, :-1]),
        ]
        for a, h in cases:
            with self.subTest(shape=(a.shape, h.shape), dtype=(a.dtype, h.dtype)):
                with self.assertRaises((TypeError, ValueError)):
                    self.cif(a, h, real_T=400)
        for value in (np.nan, np.inf, -0.1):
            a = self.alphas.copy()
            a[0, 0] = value
            with self.subTest(alpha=value), self.assertRaises(ValueError):
                self.cif(a, self.hidden, real_T=400)
        self.hidden[0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            self.cif(self.alphas, self.hidden, real_T=400)


if __name__ == "__main__":
    unittest.main()
