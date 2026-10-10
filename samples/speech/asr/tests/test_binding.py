import unittest
import numpy as np
from dataclasses import replace
from types import SimpleNamespace
from samples.speech.asr.runtime.python.cli import resolve_selection, SAMPLE_DIR
from samples.speech.asr.runtime.python.asr import bind_model
from samples.speech.asr.runtime.python.cli import load_vocabulary
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


class CanonicalStageTests(unittest.TestCase):
    """Readable-runtime canonical stage names on the ASR task."""

    @staticmethod
    def make_task(raw, vocabulary=("<pad>", "a", "b"), decode_mode="ctc"):
        meta = metadata()
        meta["output_shapes"] = {"logits": tuple(raw.shape)}
        meta["output_quants"] = {
            "logits": SimpleNamespace(
                quant_type="SCALE", scale=[1.0], zero_point=[0], axis=2
            )
        }
        binding = SimpleNamespace(
            input_name="audio",
            output_name="logits",
            metadata=SimpleNamespace(
                output_shapes=meta["output_shapes"],
                output_dtypes={"logits": raw.dtype},
                output_quants=meta["output_quants"],
            ),
        )
        calls = []

        def runner(tensors):
            calls.append(tensors)
            return raw

        return ASR(runner, binding, vocabulary, decode_mode=decode_mode), calls

    def test_canonical_stages_match_legacy_aliases(self):
        raw = np.zeros((1, 2, 3), np.int32)
        raw[0, 0, 1] = 2
        task, _ = self.make_task(raw)
        wave = np.ones(20, np.float32)
        first = task.preprocess(wave, 16000)
        second = task.pre_process(wave, 16000)
        np.testing.assert_array_equal(first.tensor, second.tensor)
        self.assertEqual(first.valid_samples, second.valid_samples)
        self.assertIs(task.infer({"audio": first.tensor}), task.forward({"audio": first.tensor}))
        self.assertEqual(task.postprocess(raw), task.post_process(raw))

    def test_predict_equals_explicit_canonical_chain(self):
        raw = np.zeros((1, 3, 3), np.int32)
        raw[0, 0, 1] = 2
        raw[0, 1, 0] = 1
        raw[0, 2, 1] = 2
        task, calls = self.make_task(raw)
        wave = np.ones(20, np.float32)
        prepared = task.preprocess(wave, 16000)
        explicit = task.postprocess(task.infer({"audio": prepared.tensor}))
        self.assertEqual(task.predict(wave, 16000), explicit)
        # CTC collapses [1, 1] but keeps tokens separated by blank 0: "aa".
        self.assertEqual(task.predict(wave, 16000), "aa")
        # One runner call per explicit infer plus one per predict.
        self.assertEqual(len(calls), 3)

    def test_infer_returns_fixture_unchanged(self):
        raw = np.zeros((1, 1, 3), np.int32)
        task, calls = self.make_task(raw)
        tensor = np.zeros((1, 30000), np.float32)
        self.assertIs(task.infer({"audio": tensor}), raw)
        self.assertEqual(calls, [{"audio": tensor}])


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

        # Default decode is the source sample behavior: repeats kept.
        task = ASR(runner, binding, vocabulary)
        self.assertEqual(
            task.predict(np.ones(20, np.float32), 16000), vocabulary[5] * 3
        )
        self.assertEqual(len(calls), 1)
        self.assertIs(task.forward(calls[0]), raw)
        legacy = ASR(runner, binding, vocabulary, decode_mode="legacy")
        self.assertEqual(legacy.post_process(raw), vocabulary[5] * 3)
        meta32 = metadata()
        meta32["output_dtypes"] = {"logits": "int32"}
        meta32["output_quants"] = {
            "logits": SimpleNamespace(
                quant_type="SCALE", scale=[1.0], zero_point=[0], axis=2
            )
        }
        binding32 = bind_model(resolve_selection("s100"), meta32)
        raw32 = np.zeros((1, 4, 3503), np.int32)
        raw32[0, 0, 0] = 16777216
        raw32[0, 0, 1] = 16777217
        raw32[0, 1:4, 5] = 1
        task32 = ASR(lambda tensors: raw32, binding32, vocabulary)
        self.assertEqual(
            task32.predict(np.ones(20, np.float32), 16000),
            vocabulary[1] + vocabulary[5] * 3,
        )
        ctc32 = ASR(lambda tensors: raw32, binding32, vocabulary, decode_mode="ctc")
        self.assertEqual(
            ctc32.post_process(raw32), vocabulary[1] + vocabulary[5]
        )
        legacy32 = ASR(lambda tensors: raw32, binding32, vocabulary, decode_mode="legacy")
        self.assertEqual(
            legacy32.post_process(raw32), vocabulary[1] + vocabulary[5] * 3
        )


class AnnotationGlobalTests(unittest.TestCase):
    """The merged modules must keep every annotation global resolvable.

    ``Binding.selection`` evaluates eagerly on Python 3.10-3.13 (no future
    annotations in asr.py), and ``AudioChunk.waveform`` must resolve to a
    real NumPy dtype; both were silent NameErrors when ``Selection``/``np``
    moved modules during consolidation.
    """

    def test_binding_and_audio_chunk_annotations_resolve_to_real_globals(self):
        import typing

        import numpy as np

        from samples.speech.asr.runtime.python import asr, cli

        hints = typing.get_type_hints(asr.Binding)
        self.assertIs(hints["selection"], cli.Selection)
        self.assertIs(typing.get_type_hints(cli.AudioChunk)["waveform"], np.ndarray)
