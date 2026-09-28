"""Pure ASR contract regressions; no file/SDK I/O inside model stages."""

import unittest
import numpy as np
from samples.speech.asr.runtime.python.decoding import (
    decode_ids,
    decode_logits,
    validate_vocabulary,
)
from samples.speech.asr.runtime.python.frontend import (
    Config,
    prepare_chunk,
    source_chunk_size,
)


class DecodeTests(unittest.TestCase):
    def test_ctc_repeat_then_blank_removal(self):
        vocab = ("<pad>", "a", "b")
        self.assertEqual(decode_ids([1, 1, 0, 1, 2, 2], vocab, "ctc"), "aab")
        self.assertEqual(decode_ids([1, 1, 0, 1, 2, 2], vocab, "legacy"), "aaabb")
        self.assertEqual(decode_ids([0, 0, 0], vocab), "")
        self.assertEqual(decode_ids([], vocab), "")

    def test_tokens_preserved_and_calls_independent(self):
        vocab = ("<pad>", "中", "|", "<unk>")
        self.assertEqual(decode_ids([1, 1, 2, 3], vocab), "中|<unk>")
        self.assertEqual(decode_ids([1], vocab) + decode_ids([1], vocab), "中中")

    def test_greedy_ties_and_very_negative_logits(self):
        vocab = ("<pad>", "a", "b")
        raw = np.array(
            [[[-3e35, -2e35, -3e35], [-3e35, -2e35, -2e35], [0, 0, 0]]], np.float32
        )
        self.assertEqual(decode_logits(raw, vocab), "a")
        self.assertEqual(decode_logits(raw, vocab, "legacy"), "aa")

    def test_invalid_vocabulary_ids_logits_modes(self):
        for vocab in ((), ("a", "b"), ("<pad>", "a", "a"), ("<pad>", 1)):
            with self.assertRaises(ValueError):
                validate_vocabulary(vocab)
        for ids in ([3], [-1], [True], [1.5]):
            with self.assertRaises(ValueError):
                decode_ids(ids, ("<pad>", "a", "b"))
        for raw in (
            np.zeros((2, 1, 3), np.float32),
            np.zeros((1, 1, 2), np.float32),
            np.full((1, 1, 3), np.nan, np.float32),
            np.zeros((1, 1, 3), np.int8),
        ):
            with self.assertRaises(ValueError):
                decode_logits(raw, ("<pad>", "a", "b"))
        with self.assertRaises(ValueError):
            decode_ids([1], ("<pad>", "a"), "unknown")


class FrontendTests(unittest.TestCase):
    def test_read_size_and_fixed_protocol(self):
        self.assertEqual(source_chunk_size(16000), 30000)
        self.assertEqual(source_chunk_size(44100), 82688)
        for rate in (0, -1, True, 16000.5):
            with self.assertRaises(ValueError):
                source_chunk_size(rate)
        with self.assertRaises(ValueError):
            prepare_chunk(np.ones(5, np.float32), 16000, Config(audio_maxlen=1))

    def test_normalize_before_padding_and_owned_tensor(self):
        wave = np.array([1, 2, 3], np.float32)
        result = prepare_chunk(wave, 16000)
        expected = (wave - wave.mean()) / np.sqrt(wave.var() + 1e-5)
        np.testing.assert_array_equal(result.tensor[0, :3], expected)
        self.assertEqual(result.tensor.shape, (1, 30000))
        self.assertEqual(result.valid_samples, 3)
        self.assertEqual(np.count_nonzero(result.tensor[0, 3:]), 0)
        self.assertFalse(np.shares_memory(result.tensor, wave))
        np.testing.assert_array_equal(
            prepare_chunk(np.full(4, 0.5, np.float32), 16000).tensor, 0
        )

    def test_stereo_mix_and_real_fourier_resampling(self):
        from scipy.signal import resample

        raw = np.stack(
            [
                np.linspace(-1, 1, 800, dtype=np.float32),
                np.linspace(0, 1, 800, dtype=np.float32),
            ],
            axis=1,
        )
        mono = raw.mean(axis=1)
        expected = resample(mono, 1600)
        expected = (expected - expected.mean()) / np.sqrt(expected.var() + 1e-5)
        result = prepare_chunk(raw, 8000)
        self.assertEqual(result.valid_samples, 1600)
        np.testing.assert_array_equal(result.tensor[0, :1600], expected)

    def test_invalid_chunk_and_no_silent_tail_drop(self):
        for value in (
            np.empty(0, np.float32),
            np.array([np.inf], np.float32),
            np.ones((1, 2, 2), np.float32),
            np.ones(2, np.int16),
            np.ones(30001, np.float32),
        ):
            with self.assertRaises(ValueError):
                prepare_chunk(value, 16000)
