"""Frontend math on a Torch-free host: CMVN parsing, LFR stacking, resampling.

The fbank itself needs Torch and is verified on the conversion/parity host
(server) against the vendored upstream frontend; these tests pin the
NumPy-visible parts and every rejection path.
"""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))

from samples.vision.lite_avatar.runtime.python import frontend

SAMPLE_DIR = Path(frontend.__file__).resolve().parents[2]
REAL_CMVN = SAMPLE_DIR / "model" / "am.mvn"


class CmvnTests(unittest.TestCase):
    def test_parses_the_pinned_am_mvn(self):
        cmvn = frontend.load_cmvn(REAL_CMVN)
        self.assertEqual(cmvn.shape, (2, frontend.FEATURE_DIM))
        self.assertEqual(cmvn.dtype, np.float32)
        self.assertTrue(np.isfinite(cmvn).all())
        self.assertFalse(np.allclose(cmvn[0], 0.0))

    def test_rejects_incomplete_or_nonfinite_files(self):
        bad = Path(self.id().replace(".", "_"))
        self.addCleanup(bad.unlink)
        for text in ("", "<AddShift>\n  <LearnRateCoef> 3 a b </LearnRateCoef>\n"):
            bad.write_text(text, encoding="utf-8")
            with self.assertRaises(ValueError):
                frontend.load_cmvn(bad)

    def test_apply_cmvn_uses_add_then_multiply_order(self):
        features = np.zeros((3, 4), np.float64)
        cmvn = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], np.float32)
        out = frontend.apply_cmvn(features, cmvn)
        # (0 + means) * vars, computed in float32.
        self.assertEqual(out.dtype, np.float32)
        self.assertTrue(np.allclose(out, np.tile([5.0, 12.0, 21.0, 32.0], (3, 1))))
        self.assertFalse(np.allclose(out, np.tile((0 - cmvn[0]) / cmvn[1], (3, 1))))
        # The input matrix must not be mutated in place.
        self.assertTrue(np.all(features == 0))

    def test_apply_cmvn_rejects_narrow_coefficients(self):
        with self.assertRaises(ValueError):
            frontend.apply_cmvn(np.zeros((2, 560)), np.zeros((2, 4), np.float32))


class LfrTests(unittest.TestCase):
    def test_geometry_matches_the_upstream_formula(self):
        for frames, stacked in ((1, 1), (6, 1), (7, 2), (100, 17), (2394, 399), (2395, 400)):
            with self.subTest(frames=frames):
                mat = np.arange(frames * 3, dtype=np.float32).reshape(frames, 3)
                out = frontend.apply_lfr(mat)
                self.assertEqual(out.shape, (stacked, 3 * frontend.LFR_M))
                self.assertEqual(out.dtype, np.float32)

    def test_left_padding_repeats_the_first_frame(self):
        mat = np.array([[10.0, 11.0, 12.0], [13.0, 14.0, 15.0]], np.float32)
        out = frontend.apply_lfr(mat, lfr_m=5, lfr_n=2)
        # (5-1)//2 = 2 left-padded copies of frame 0 precede the real frames.
        self.assertEqual(out.shape, (1, 15))
        expected = np.concatenate([mat[0], mat[0], mat[0], mat[1], mat[-1]])
        self.assertTrue(np.allclose(out[0], expected))

    def test_final_truncated_window_pads_with_the_last_frame(self):
        mat = np.arange(3 * 2, dtype=np.float32).reshape(3, 2)
        out = frontend.apply_lfr(mat, lfr_m=4, lfr_n=2)
        # One left-padded copy of frame 0 precedes the frames, so window 2
        # (start 2) sees [f1, f2] and repeats the final frame twice more.
        expected = np.concatenate([mat[1], mat[2], mat[2], mat[2]])
        self.assertTrue(np.allclose(out[1], expected))

    def test_rejects_empty_input(self):
        for bad in (np.zeros((0, 3)), np.zeros(3)):
            with self.assertRaises(ValueError):
                frontend.apply_lfr(bad)


class WaveformPreparationTests(unittest.TestCase):
    def test_mono_passthrough_is_float32_and_owned(self):
        wave = np.linspace(-0.5, 0.5, 400, dtype=np.float64)
        expected = wave.astype(np.float32)
        out = frontend.prepare_waveform(wave, 16000)
        self.assertEqual(out.dtype, np.float32)
        self.assertTrue(np.array_equal(out, expected))
        # The returned buffer is owned: mutating the input afterwards must
        # not change it.
        wave[:] = 9.0
        self.assertTrue(np.array_equal(out, expected))

    def test_stereo_takes_the_first_channel_after_resample(self):
        # 48 kHz stereo whose channels differ; output must track channel 0.
        time = np.arange(24000, dtype=np.float64) / 24000.0
        stereo = np.stack([np.sin(2 * np.pi * 440 * time), np.zeros(24000)], axis=1)
        out = frontend.prepare_waveform(stereo, 48000)
        self.assertEqual(out.ndim, 1)
        self.assertEqual(out.dtype, np.float32)
        # Resampled length ~ 24000 * 16000/48000 = 8000.
        self.assertTrue(7900 < out.shape[0] < 8100)

    def test_resample_uses_the_polyphase_ratio_then_first_channel(self):
        from scipy.signal import resample_poly

        time = np.arange(16000, dtype=np.float64) / 16000.0
        mono = np.sin(2 * np.pi * 220 * time)
        stereo = np.stack([mono, -mono], axis=1)
        out = frontend.prepare_waveform(stereo, 48000)
        # Same delegation as the source loader: polyphase 48k -> 16k on all
        # channels, float32 cast, then the first channel only.
        expected = resample_poly(stereo, 1, 3)[:, 0].astype(np.float32)
        self.assertEqual(out.dtype, np.float32)
        np.testing.assert_allclose(out, expected, atol=1e-6)

    def test_rejects_bad_rate_short_or_nonfinite_audio(self):
        wave = np.zeros(400, np.float32)
        for rate in (0, -16000, 16000.5, True, None):
            with self.subTest(rate=rate), self.assertRaises(ValueError):
                frontend.prepare_waveform(wave, rate)
        wave[10] = np.inf
        with self.assertRaises(ValueError):
            frontend.prepare_waveform(wave, 16000)
        for bad in (np.zeros(399, np.float32), np.zeros((0,), np.float32), np.zeros((0, 2))):
            with self.subTest(bad=bad.shape), self.assertRaises(ValueError):
                frontend.prepare_waveform(bad, 16000)


class FrontendClassTests(unittest.TestCase):
    def test_pins_the_digest_of_the_bundled_am_mvn(self):
        real = frontend.AvatarFrontend(REAL_CMVN)
        self.assertEqual(real.cmvn.shape[1], frontend.FEATURE_DIM)
        tampered = Path(self.id().replace(".", "_"))
        self.addCleanup(tampered.unlink)
        tampered.write_bytes(REAL_CMVN.read_bytes() + b"\n0.0 0.0\n")
        with self.assertRaises(ValueError):
            frontend.AvatarFrontend(tampered)

    def test_features_from_mono_validates_the_prepared_audio(self):
        real = frontend.AvatarFrontend(REAL_CMVN)
        for bad in (
            np.zeros((2, 100), np.float32),
            np.zeros(100, np.float64),
            np.zeros(frontend.MIN_SAMPLES - 1, np.float32),
            np.array([np.nan] * frontend.MIN_SAMPLES, np.float32),
            "not audio",
        ):
            with self.subTest(bad=type(bad)), self.assertRaises(ValueError):
                real.features_from_mono(bad)


if __name__ == "__main__":
    unittest.main()
