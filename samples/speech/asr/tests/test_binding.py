import unittest
import numpy as np
from dataclasses import replace
from types import SimpleNamespace
from samples.speech.asr.runtime.python.model_binding import (
    resolve_selection,
    bind_model,
    SAMPLE_DIR,
)
from samples.speech.asr.runtime.python.vocabulary import load_vocabulary
from samples.speech.asr.runtime.python.asr import ASR


def metadata():
    return dict(
        model_name="asr",
        model_names=["asr"],
        input_names=["audio"],
        output_names=["logits"],
        input_shapes={"audio": [1, 30000]},
        output_shapes={"logits": [1, 4, 3503]},
        input_dtypes={"audio": "float32"},
        output_dtypes={"logits": "float32"},
    )


class BindingTests(unittest.TestCase):
    def test_exact_targets_and_no_external_path_guess(self):
        for target in ("s100", "s600"):
            self.assertEqual(
                resolve_selection(target).asset.reference, f"s:asr:{target}/asr.hbm"
            )
        for target in ("x5", "s100p"):
            with self.assertRaises(ValueError):
                resolve_selection(target)
        with self.assertRaises(ValueError):
            resolve_selection("s600", asset_id="s:asr:s100/asr.hbm")
        with self.assertRaises(ValueError):
            resolve_selection("s100", model_path="unknown.hbm")

    def test_fixed_input_and_vocabulary_width(self):
        selection = resolve_selection("s100")
        self.assertEqual(bind_model(selection, metadata()).output_name, "logits")
        for key, value in [
            ("input_shapes", {"audio": [1, 30001]}),
            ("output_shapes", {"logits": [1, 4, 3]}),
            ("input_dtypes", {"audio": "int16"}),
            ("output_names", ["a", "b"]),
        ]:
            meta = metadata()
            meta[key] = value
            with self.assertRaises(ValueError):
                bind_model(selection, meta)
        with self.assertRaises(ValueError):
            bind_model(replace(selection, target="s600"), metadata())

    def test_stages_and_integer_logits(self):
        vocabulary = load_vocabulary(SAMPLE_DIR / "test_data/vocab.json")
        self.assertEqual(len(vocabulary), 3503)
        self.assertEqual(vocabulary[0], "<pad>")
        meta = metadata()
        meta["output_dtypes"] = {"logits": "int8"}
        with self.assertRaises(ValueError):
            bind_model(resolve_selection("s100"), meta)
        meta["output_quants"] = {
            "logits": SimpleNamespace(
                quant_type="SCALE", scale=[0.1], zero_point=[0], axis=0
            )
        }
        binding = bind_model(resolve_selection("s100"), meta)
        raw = np.zeros((1, 4, 3503), np.int8)
        raw[0, 0:2, 5] = 10
        raw[0, 3, 5] = 10
        calls = []

        def runner(tensors):
            calls.append(tensors)
            return raw

        task = ASR(runner, binding, vocabulary)
        self.assertEqual(
            task.predict(np.ones(20, np.float32), 16000), vocabulary[5] * 2
        )
        self.assertEqual(len(calls), 1)
        self.assertIs(task.forward(calls[0]), raw)
        legacy = ASR(runner, binding, vocabulary, decode_mode="legacy")
        self.assertEqual(legacy.post_process(raw), vocabulary[5] * 3)
