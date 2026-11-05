"""Regression checks for upstream XLSR normalization and valid mask geometry."""
import unittest
from unittest.mock import patch
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

    def test_decode_uses_valid_frames_and_accounts_for_a_tiny_tail(self):
        from samples.speech.asr.runtime.python.public import transcribe
        vocab=['<pad>']+[f'token{i}'for i in range(1,3503)]
        logits=np.full((1,93,3503),-1,np.float32)
        logits[:,:,2]=1;logits[:,0,1]=2
        calls=[]
        def runner(feed):
            calls.append(feed)
            return {'logits':logits}
        windows=[(np.ones(400,np.float32),{'source_start':0,'source_samples':400,'source_rate':16000}),
                 (np.ones(7,np.float32),{'source_start':400,'source_samples':7,'source_rate':16000})]
        with patch('samples.speech.asr.runtime.python.public.audio_windows',return_value=windows):
            result=transcribe('unused.wav',runner,vocab)
        self.assertEqual(result['text'],'token1')
        self.assertEqual(result['source_samples'],407)
        self.assertEqual(len(calls),1)
        self.assertEqual(result['chunks'][1]['inference'],'no-valid-feature-frame')
        np.testing.assert_array_equal(calls[0]['attention_mask'][0,400:],0)

    def test_runtime_rejects_partial_or_nonfinite_logits(self):
        from samples.speech.asr.runtime.python.public import transcribe
        windows=[(np.ones(400,np.float32),{'source_start':0,'source_samples':400,'source_rate':16000})]
        for value in [np.zeros((1,93,3502),np.float32),np.full((1,93,3503),np.nan,np.float32)]:
            with patch('samples.speech.asr.runtime.python.public.audio_windows',return_value=windows):
                with self.assertRaises(ValueError):transcribe('unused.wav',lambda _: {'logits':value},[])
