"""Pure ASR contract regressions; no file/SDK I/O inside model stages."""

import unittest
from types import SimpleNamespace
import numpy as np
from samples.speech.asr.runtime.python.decoding import (
    decode_exact_logits,
    decode_ids,
    decode_logits,
    validate_vocabulary,
)
from samples.speech.asr.runtime.python.frontend import (
    Config,
    prepare_chunk,
    source_chunk_size,
)
from samples.speech.asr.runtime.python.postprocess import transcribe


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


class IntegerTranscribeTests(unittest.TestCase):
    """Integer SCALE logits keep their affine ordering through argmax (ASR-R1)."""

    @staticmethod
    def binding(dtype, shape, scale, zero_point, axis=2):
        quant = SimpleNamespace(
            quant_type="SCALE", scale=scale, zero_point=zero_point, axis=axis
        )
        return SimpleNamespace(
            output_name="logits",
            metadata=SimpleNamespace(
                output_shapes={"logits": shape},
                output_dtypes={"logits": dtype},
                output_quants={"logits": quant},
            ),
        )

    def test_reviewed_int32_counterexample_decodes_token1(self):
        vocabulary = ("<pad>",) + tuple(f"token{i}" for i in range(1, 3503))
        raw = np.zeros((1, 1, 3503), np.int32)
        raw[0, 0, 0] = 16777216
        raw[0, 0, 1] = 16777217
        # The reviewed false tie: float32 cannot represent 2**24 + 1.
        self.assertEqual(raw[0, 0, :2].astype(np.float32)[0], raw[0, 0, :2].astype(np.float32)[1])
        binding = self.binding("int32", (1, 1, 3503), [1.0], [0])
        self.assertEqual(transcribe(raw, binding, vocabulary, "ctc"), "token1")
        self.assertEqual(transcribe(raw, binding, vocabulary, "legacy"), "token1")

    def test_int32_per_channel_scale_and_offset_ranking(self):
        vocabulary = ("<pad>", "a", "b")
        scales = self.binding("int32", (1, 1, 3), [1.0, 3.0, 1.0], [0, 0, 0])
        raw = np.array([[[0, 4, 5]]], np.int32)
        # Dequantized [0, 12, 5] wins although the raw argmax favors "b".
        self.assertEqual(transcribe(raw, scales, vocabulary, "ctc"), "a")
        offsets = self.binding("int32", (1, 1, 3), [2.0, 1.0, 1.0], [9, 0, 0])
        raw = np.array([[[6, 2, 3]]], np.int32)
        # Dequantized [-6, 2, 3] wins although the raw argmax favors blank.
        self.assertEqual(transcribe(raw, offsets, vocabulary, "ctc"), "b")

    def test_int32_true_ties_keep_lowest_id(self):
        vocabulary = ("<pad>", "a", "b")
        tie_with_blank = self.binding("int32", (1, 1, 3), [1.0, 2.0, 1.0], [0, 0, 0])
        raw = np.array([[[2, 1, 0]]], np.int32)
        # Dequantized [2, 2, 0]: genuinely equal scores tie to the lowest ID.
        self.assertEqual(transcribe(raw, tie_with_blank, vocabulary, "ctc"), "")
        tie_between_tokens = self.binding("int32", (1, 1, 3), [1.0, 1.0, 1.0], [0, 0, 0])
        raw = np.array([[[0, 2, 2]]], np.int32)
        self.assertEqual(transcribe(raw, tie_between_tokens, vocabulary, "legacy"), "a")

    def test_integer_logits_follow_selected_decode_mode(self):
        vocabulary = ("<pad>", "a", "b")
        binding = self.binding("int32", (1, 4, 3), [1.0], [0])
        raw = np.zeros((1, 4, 3), np.int32)
        raw[0, :2, 1] = 3
        raw[0, 3, 1] = 3
        self.assertEqual(transcribe(raw, binding, vocabulary, "ctc"), "aa")
        self.assertEqual(transcribe(raw, binding, vocabulary, "legacy"), "aaa")

    def test_float32_transcribe_keeps_f32_decoding_and_ignores_vestigial_quant(self):
        vocabulary = ("<pad>", "a", "b")
        binding = self.binding("float32", (1, 1, 3), [0.5], [3])
        raw = np.array([[[-3e35, -2e35, -3e35]]], np.float32)
        self.assertEqual(transcribe(raw, binding, vocabulary, "ctc"), "a")
        self.assertEqual(
            transcribe(np.zeros((1, 1, 3), np.float32), binding, vocabulary, "ctc"), ""
        )

    def test_exact_decoder_rejects_non_exact_inputs(self):
        vocabulary = ("<pad>", "a", "b")
        for raw in (
            np.zeros((1, 1, 3), np.float32),
            np.zeros((1, 1, 3), np.int32),
            np.full((1, 1, 3), np.nan, np.float64),
            np.zeros((2, 1, 3), np.float64),
            np.zeros((1, 1, 2), np.float64),
        ):
            with self.assertRaises(ValueError):
                decode_exact_logits(raw, vocabulary)


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
