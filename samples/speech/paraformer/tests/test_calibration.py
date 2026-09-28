"""Calibration selection and bridge failures; never hardware or synthetic calibration acceptance."""

from pathlib import Path
import tempfile
import unittest
import numpy as np


class Calibration(unittest.TestCase):
    def test_wav_selection_is_sorted_nonempty_and_positive(self):
        from samples.speech.paraformer.conversion.calibration import select_wavs

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(ValueError):
                select_wavs(root, 50)
            (root / "b").mkdir()
            for relative in ("z.wav", "b/a.wav", "ignored.txt"):
                (root / relative).write_bytes(b"fixture path only")
            self.assertEqual(select_wavs(root, 1), ((root / "b/a.wav").resolve(),))
            self.assertEqual(len(select_wavs(root, 50)), 2)
            for count in (0, -1, True):
                with self.assertRaises(ValueError):
                    select_wavs(root, count)

    def test_bridge_reuses_unmasked_cif_and_preserves_named_inputs(self):
        from samples.speech.paraformer.conversion.calibration import intermediates
        from samples.speech.paraformer.runtime.python.cif import cif_numpy
        from samples.speech.paraformer.runtime.python.model_binding import CONTEXT

        context = np.ones((1, 400, 512), np.float32)
        alphas = np.full((1, 401), 0.3, np.float32)
        hidden = np.ones((1, 401, 512), np.float32)
        seen = []

        def encoder(feed):
            seen.append(set(feed))
            return {CONTEXT: context}

        def predictor(feed):
            seen.append(set(feed))
            return {
                "/predictor/Add_output_0": alphas,
                "/predictor/Concat_5_output_0": hidden,
            }

        arrays = intermediates(np.zeros((1, 400, 560), np.float32), encoder, predictor)
        expected, count = cif_numpy(alphas, hidden, real_T=None)
        self.assertEqual(seen, [{"speech"}, {CONTEXT}])
        self.assertEqual(
            set(arrays),
            {
                "speech",
                "encoder_after_norm_Add_1_output_0",
                "predictor_Add_output_0",
                "predictor_Concat_5_output_0",
                "shape_8609",
                "token_num",
                "bias_embed",
            },
        )
        np.testing.assert_array_equal(arrays["shape_8609"], expected)
        np.testing.assert_array_equal(arrays["token_num"], count)
        self.assertEqual(arrays["token_num"].dtype, np.int32)
        self.assertTrue(np.all(arrays["bias_embed"] == 0))

    def test_invalid_features_or_intermediate_are_not_silently_skipped(self):
        from samples.speech.paraformer.conversion.calibration import intermediates
        from samples.speech.paraformer.runtime.python.model_binding import CONTEXT

        good = np.zeros((1, 400, 560), np.float32)
        for bad in (good.astype(np.float64), good[:, :20], good + np.nan):
            with self.assertRaises(ValueError):
                intermediates(
                    bad, lambda feed: self.fail("must validate before inference"), None
                )
        for bad in (
            np.zeros((1, 400, 512), np.float64),
            np.zeros((1, 1, 512), np.float32),
        ):
            with self.assertRaises(ValueError):
                intermediates(good, lambda feed: {CONTEXT: bad}, None)


if __name__ == "__main__":
    unittest.main()
