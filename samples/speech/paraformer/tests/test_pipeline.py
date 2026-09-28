"""Exercise actual CPU orchestration with explicit synthetic model boundaries."""

import importlib
import importlib.util
import unittest

import numpy as np


class PipelineTests(unittest.TestCase):
    def setUp(self):
        name = "samples.speech.paraformer.runtime.python.pipeline"
        self.assertIsNotNone(
            importlib.util.find_spec(name), "Missing Paraformer pipeline"
        )
        self.module = importlib.import_module(name)
        self.vocabulary = [f"t{i}" for i in range(8404)]
        self.vocabulary[:6] = ["<blank>", "<s>", "</s>", "中", "文@@", "好"]
        self.features = np.zeros((1, 400, 560), np.float32)
        self.features[0, 0, 0] = 7
        self.calls = []

    def make_pipeline(self, *, empty=False, malformed=False):
        def encoder(inputs):
            self.calls.append("encoder")
            self.assertEqual(set(inputs), {"speech"})
            self.assertEqual(inputs["speech"][0, 0, 0], 7)
            return {"context": np.full((1, 400, 512), 3, np.float32)}

        def predictor(inputs):
            self.calls.append("predictor")
            self.assertEqual(set(inputs), {"context"})
            self.assertTrue(np.all(inputs["context"] == 3))
            alphas = np.zeros((1, 401), np.float32)
            if not empty:
                alphas[0, :4] = 1
            alphas[0, 400] = 1  # Padding must never create a fifth token.
            return {
                "alphas": alphas,
                "hidden": np.full((1, 401, 512), 2, np.float32),
            }

        def decoder(inputs):
            self.calls.append("decoder")
            self.assertEqual(set(inputs), {"context", "token_num", "bias", "acoustic"})
            self.assertTrue(np.all(inputs["context"] == 3))
            np.testing.assert_array_equal(inputs["token_num"], [4])
            self.assertEqual(inputs["token_num"].dtype, np.int32)
            self.assertTrue(np.all(inputs["acoustic"][0, :4] == 2))
            self.assertFalse(inputs["acoustic"][0, 4:].any())
            self.assertEqual(inputs["bias"].shape, (1, 1, 512))
            self.assertFalse(inputs["bias"].any())
            logits = np.zeros((1, 100, 8404), np.float32)
            logits[0, np.arange(4), [3, 3, 4, 2]] = 1
            if malformed:
                logits[0, 0, 0] = np.nan
            return {"logits": logits}

        names = self.module.TensorNames(
            encoder_input="speech",
            encoder_output="context",
            predictor_input="context",
            predictor_alphas="alphas",
            predictor_hidden="hidden",
            decoder_context="context",
            decoder_count="token_num",
            decoder_bias="bias",
            decoder_acoustic="acoustic",
            decoder_logits="logits",
        )
        return self.module.ParaformerPipeline(
            encoder, predictor, decoder, names, self.vocabulary
        )

    def test_model_order_masking_and_source_text_rules(self):
        pipeline = self.make_pipeline()
        result = pipeline.predict(self.features, 4)
        self.assertEqual(self.calls, ["encoder", "predictor", "decoder"])
        self.assertEqual(result.text, "中中文")  # No CTC repeat collapse.
        self.assertEqual(result.token_ids, (3, 3, 4, 2))
        self.assertEqual(result.token_count, 4)
        self.assertTrue(result.decoder_executed)
        self.assertEqual(
            set(result.timings_ms), {"encoder", "predictor", "cif", "decoder"}
        )
        self.assertTrue(all(x >= 0 for x in result.timings_ms.values()))
        self.assertEqual(self.features[0, 0, 0], 7)

    def test_no_token_skips_decoder_and_returns_empty_text(self):
        result = self.make_pipeline(empty=True).predict(self.features, 4)
        self.assertEqual(self.calls, ["encoder", "predictor"])
        self.assertEqual(result.text, "")
        self.assertEqual(result.token_ids, ())
        self.assertEqual(result.token_count, 0)
        self.assertFalse(result.decoder_executed)
        self.assertIsNone(result.timings_ms["decoder"])

    def test_bad_frontend_contract_fails_before_any_model(self):
        pipeline = self.make_pipeline()
        for features, length in (
            (self.features.astype(np.float64), 4),
            (self.features[:, :-1], 4),
            (self.features, 0),
            (self.features, 401),
            (self.features, True),
        ):
            with self.subTest(shape=features.shape, length=length), self.assertRaises(
                ValueError
            ):
                pipeline.predict(features, length)
        self.assertEqual(self.calls, [])

    def test_bad_decoder_output_is_not_a_transcript(self):
        with self.assertRaises(ValueError):
            self.make_pipeline(malformed=True).predict(self.features, 4)

    def test_vocabulary_must_match_8404_ordered_nonempty_tokens(self):
        for vocabulary in (self.vocabulary[:-1], {0: "x"}, [""] * 8404):
            self.vocabulary = vocabulary
            with self.assertRaises(ValueError):
                self.make_pipeline()

    def test_missing_predictor_output_stops_before_decoder(self):
        pipeline = self.make_pipeline()
        pipeline.predictor = lambda inputs: {"wrong": np.zeros((1, 401), np.float32)}
        with self.assertRaises(ValueError):
            pipeline.predict(self.features, 4)
        self.assertEqual(self.calls, ["encoder"])


if __name__ == "__main__":
    unittest.main()
