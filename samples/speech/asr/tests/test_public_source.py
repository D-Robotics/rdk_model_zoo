"""Regression checks for upstream XLSR normalization and valid mask geometry."""
import unittest
import numpy as np
from samples.speech.asr.runtime.python.public_preprocess import feature_frames, prepare_waveform


class PublicSourceTests(unittest.TestCase):
    def test_upstream_epsilon_and_normalize_before_padding(self):
        raw = np.tile(np.asarray([1e-4, -1e-4], np.float32), 200)
        value = prepare_waveform(raw)
        expected = .30151134457776363
        np.testing.assert_allclose(value['waveform'][0, :400:2], expected, atol=1e-7, rtol=0)
        np.testing.assert_array_equal(value['waveform'][0, 400:], 0)
        np.testing.assert_array_equal(value['attention_mask'][0, :400], 1)
        np.testing.assert_array_equal(value['attention_mask'][0, 400:], 0)
        self.assertEqual(value['attention_mask'].dtype, np.float32)

    def test_frame_geometry_includes_receptive_field_boundary(self):
        self.assertEqual(feature_frames(399), 0)
        self.assertEqual(feature_frames(400), 1)
        self.assertEqual(feature_frames(30000), 93)

    def test_window_boundary_does_not_silently_drop_audio(self):
        with self.assertRaises(ValueError):
            prepare_waveform(np.zeros(30001, np.float32))

    def test_pcm_integer_samples_require_explicit_conversion(self):
        with self.assertRaises(ValueError):
            prepare_waveform(np.zeros(400, np.int16))
