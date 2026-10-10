"""Stage APIs preserve raw values, owned call state and error attribution."""

import unittest
from unittest.mock import Mock, patch
import numpy as np
from samples.speech.paraformer.tests import test_pipeline as fixtures


class StageAPI(unittest.TestCase):
    def fixture(self):
        fixture = fixtures.PipelineTests()
        fixture.setUp()
        return fixture, fixture.make_pipeline()

    def test_explicit_stages_match_pipeline(self):
        from samples.speech.paraformer.runtime.python.cif import cif_numpy

        fixture, pipeline = self.fixture()
        result = pipeline.predict(fixture.features, 4)
        enc, pred, dec = (
            pipeline.encoder_stage,
            pipeline.predictor_stage,
            pipeline.decoder_stage,
        )
        prepared = enc.pre_process(fixture.features)
        context = enc.post_process(enc.forward(prepared.tensors))
        prepared = pred.pre_process(context)
        alphas, hidden = pred.post_process(pred.forward(prepared.tensors))
        acoustic, count = cif_numpy(alphas, hidden, real_T=4)
        prepared = dec.pre_process(context, count, acoustic)
        decoded = dec.post_process(dec.forward(prepared.tensors), prepared.context)
        self.assertEqual(
            (decoded.text, decoded.token_ids, decoded.token_count),
            (result.text, result.token_ids, result.token_count),
        )

    def test_raw_forward_is_not_decoding_or_activation(self):
        _, pipeline = self.fixture()
        for name in ("encoder", "predictor", "decoder"):
            with self.subTest(stage=name):
                task = getattr(pipeline, name + "_stage")
                raw = {
                    key: np.full(shape, -7.5, dtype)
                    for key, (shape, dtype) in task.outputs.items()
                }
                runner = Mock(return_value=raw)
                setattr(pipeline, name, runner)
                feed = {
                    key: np.zeros(shape, dtype)
                    for key, (shape, dtype) in task.inputs.items()
                }
                with patch.object(
                    task, "post_process", side_effect=AssertionError("not raw")
                ):
                    returned = task.forward(feed)
                for key in raw:
                    np.testing.assert_array_equal(returned[key], raw[key])
                    self.assertFalse(np.shares_memory(returned[key], raw[key]))
                runner.assert_called_once()

    def test_cif_failure_is_attributed_before_decoder(self):
        from samples.speech.paraformer.runtime.python.pipeline import StageError

        fixture, pipeline = self.fixture()
        pipeline.decoder = Mock()
        original = ValueError("invalid integration weights")
        with patch(
            "samples.speech.paraformer.runtime.python.pipeline.cif_numpy",
            side_effect=original,
        ):
            with self.assertRaises(StageError) as caught:
                pipeline.predict(fixture.features, 4)
        self.assertEqual(caught.exception.stage, "cif")
        self.assertIs(caught.exception.__cause__, original)
        pipeline.decoder.assert_not_called()

    def test_explicit_decoder_forward_rejects_out_of_range_count(self):
        from samples.speech.paraformer.runtime.python.pipeline import StageError

        _, pipeline = self.fixture()
        task = pipeline.decoder_stage
        pipeline.decoder = Mock()
        feed = {
            key: np.zeros(shape, dtype) for key, (shape, dtype) in task.inputs.items()
        }
        feed[task.count_name][0] = 101
        with self.assertRaisesRegex(StageError, "decoder forward"):
            task.forward(feed)
        pipeline.decoder.assert_not_called()

    def test_decoder_context_and_inputs_do_not_leak_between_calls(self):
        _, pipeline = self.fixture()
        task = pipeline.decoder_stage
        context = np.zeros((1, 400, 512), np.float32)
        acoustic = np.zeros((1, 100, 512), np.float32)
        a = task.pre_process(context, np.array([2], np.int32), acoustic)
        b = task.pre_process(context, np.array([4], np.int32), acoustic)
        context.fill(99)
        acoustic.fill(99)
        self.assertEqual(a.context, 2)
        self.assertEqual(b.context, 4)
        self.assertFalse(a.tensors["context"].any())
        self.assertFalse(b.tensors["acoustic"].any())
        logits = np.zeros((1, 100, 8404), np.float32)
        logits[:, :, 3] = 1
        for prepared, expected in ((a, "中中"), (b, "中中中中"), (a, "中中")):
            self.assertEqual(
                task.post_process({"logits": logits}, prepared.context).text, expected
            )

    def test_stage_error_retains_original_cause_and_stops_later_models(self):
        from samples.speech.paraformer.runtime.python.pipeline import StageError

        fixture, pipeline = self.fixture()
        original = RuntimeError("transport failed")
        pipeline.predictor = Mock(side_effect=original)
        pipeline.decoder = Mock()
        with self.assertRaises(StageError) as caught:
            pipeline.predict(fixture.features, 4)
        self.assertEqual(caught.exception.stage, "predictor")
        self.assertEqual(caught.exception.operation, "forward")
        self.assertIs(caught.exception.__cause__, original)
        pipeline.decoder.assert_not_called()

    def test_malformed_outputs_identify_stage(self):
        from samples.speech.paraformer.runtime.python.pipeline import StageError

        fixture, pipeline = self.fixture()
        pipeline.encoder = lambda feed: {"context": np.zeros((1, 1, 512), np.float32)}
        with self.assertRaisesRegex(StageError, "encoder forward"):
            pipeline.predict(fixture.features, 4)


if __name__ == "__main__":
    unittest.main()
