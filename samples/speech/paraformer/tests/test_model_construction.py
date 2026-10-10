"""Model-owned loading preserves numerical connections and SDK scheduling."""

from types import SimpleNamespace
import unittest

import numpy as np

from samples.speech.paraformer.runtime.python.cli import resolve_selections
from samples.speech.paraformer.runtime.python.pipeline import ParaformerPipeline
from samples.speech.paraformer.tests.test_binding import metadata


class ModelConstructionTests(unittest.TestCase):
    def test_from_models_preserves_three_model_values_and_scheduling(self):
        # Missing loading or wrong stage wiring must fail before text decoding.
        self.assertTrue(callable(getattr(ParaformerPipeline, "from_models", None)))
        runtimes = {}
        context_name = "/encoder/after_norm/Add_1_output_0"

        def factory(path):
            stage = next(s for s in ("encoder", "predictor", "decoder") if s in str(path))
            meta = metadata(stage)
            sdk = SimpleNamespace(model_names=[stage], calls=[], scheduling=[])
            for field in ("input_names", "input_shapes", "input_dtypes",
                          "output_names", "output_shapes", "output_dtypes"):
                setattr(sdk, field, {stage: getattr(meta, field)})
            sdk.set_scheduling_params = lambda **kw: sdk.scheduling.append(kw)

            def run(feed):
                inputs = feed[stage]
                sdk.calls.append({key: value.copy() for key, value in inputs.items()})
                outputs = {key: np.zeros(shape, dtype=meta.output_dtypes[key])
                           for key, shape in meta.output_shapes.items()}
                if stage == "encoder":
                    outputs[context_name][0, :, :] = inputs["speech"][0, :, :1]
                elif stage == "predictor":
                    outputs["/predictor/Add_output_0"][0, :2] = 1
                    outputs["/predictor/Concat_5_output_0"][0, :400] = inputs[context_name][0]
                else:
                    outputs["logits"][0, 0, 3] = 2
                    outputs["logits"][0, 1, 4] = 4
                    outputs["token_num"][:] = 2
                return {stage: outputs}

            sdk.run = run
            runtimes[stage] = sdk
            return sdk

        model = ParaformerPipeline.from_models(
            resolve_selections("s100"), [f"token{i}" for i in range(8404)],
            runtime_factory=factory,
        )
        model.set_scheduling_params(priority=7, bpu_cores=[0])
        features = np.zeros((1, 400, 560), np.float32)
        features[0, 0, 0], features[0, 1, 0] = 2, 4
        result = model.predict(features, 2)
        self.assertEqual(result.text, "token3token4")
        self.assertEqual(result.token_ids, (3, 4))
        self.assertEqual(result.token_count, 2)
        np.testing.assert_array_equal(runtimes["encoder"].calls[0]["speech"], features)
        np.testing.assert_array_equal(
            runtimes["predictor"].calls[0][context_name][0, :2, 0], [2, 4])
        decoder_feed = runtimes["decoder"].calls[0]
        np.testing.assert_array_equal(decoder_feed[context_name][0, :2, 0], [2, 4])
        np.testing.assert_array_equal(decoder_feed["onnx::Shape_8609"][0, :2, 0], [2, 4])
        np.testing.assert_array_equal(decoder_feed["token_num"], np.array([2], np.int32))
        self.assertEqual(decoder_feed["bias_embed"].shape, (1, 1, 512))
        self.assertFalse(decoder_feed["bias_embed"].any())
        self.assertEqual(tuple(r.binding.selection.stage for r in model.runners),
                         ("encoder", "predictor", "decoder"))
        for stage, sdk in runtimes.items():
            self.assertEqual(len(sdk.calls), 1)
            self.assertEqual(sdk.scheduling,
                             [{"priority": {stage: 7}, "bpu_cores": {stage: [0]}}])
        runtimes["decoder"].set_scheduling_params = None
        with self.assertRaises(RuntimeError):
            model.set_scheduling_params(priority=8)
        self.assertEqual(len(runtimes["encoder"].scheduling), 1)
