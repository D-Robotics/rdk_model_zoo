"""Host input contracts, independent of optional Torch/FunASR packages."""

import importlib
import importlib.util
import unittest

import numpy as np


class FrontendContractTests(unittest.TestCase):
    def setUp(self):
        name = "samples.speech.paraformer.runtime.python.frontend"
        self.assertIsNotNone(importlib.util.find_spec(name), "Missing audio frontend")
        self.module = importlib.import_module(name)

    def test_stereo_mean_is_float32_owned_and_keeps_sample_count(self):
        audio = np.tile(np.array([[0.25, 0.75]], dtype=np.float32), (400, 1))
        mono = self.module.prepare_waveform(audio, 16000)
        np.testing.assert_array_equal(mono, np.full(400, 0.5, np.float32))
        self.assertEqual(mono.dtype, np.float32)
        self.assertFalse(np.shares_memory(mono, audio))

    def test_mono_input_is_owned_without_normalization(self):
        audio = np.linspace(-2, 2, 500, dtype=np.float32)
        actual = self.module.prepare_waveform(audio, 16000)
        np.testing.assert_array_equal(actual, audio)
        self.assertFalse(np.shares_memory(actual, audio))

    def test_invalid_audio_fails_without_resampling_or_implicit_cast(self):
        cases = [
            (np.zeros(500, np.float64), 16000),
            (np.zeros(500, np.float32), 8000),
            (np.zeros(0, np.float32), 16000),
            (np.zeros((500, 0), np.float32), 16000),
            (np.zeros((1, 500, 1), np.float32), 16000),
            (np.full(500, np.nan, np.float32), 16000),
        ]
        for audio, rate in cases:
            with self.subTest(shape=audio.shape, rate=rate), self.assertRaises(
                ValueError
            ):
                self.module.prepare_waveform(audio, rate)


if __name__ == "__main__":
    unittest.main()
