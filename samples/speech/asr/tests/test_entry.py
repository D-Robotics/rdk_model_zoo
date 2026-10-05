"""Entry readability: per-chunk predict with opt-in per-call details.

The entry loop composes one ``ASR.predict`` per audio chunk instead of
spelling the three stages in ``main``; the streaming report's per-chunk
geometry comes from the same single prediction through the opt-in details
record, so each chunk is still executed exactly once.
"""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from samples.speech.asr.runtime.python import asr as asr_module
from samples.speech.asr.runtime.python.asr import ASR
from samples.speech.asr.tests.test_runtime import FakeRuntime


def make_task():
    """One ASR over a tiny 3-token binding and a counting raw runner."""
    binding = SimpleNamespace(
        input_name="audio",
        output_name="logits",
        metadata=SimpleNamespace(
            output_shapes={"logits": (1, 4, 3)},
            output_dtypes={"logits": "float32"},
            output_quants={},
        ),
    )
    calls = []

    def runner(tensors):
        calls.append(tensors)
        raw = np.zeros((1, 4, 3), np.float32)
        raw[0, :2, 1] = 1
        raw[0, 3, 1] = 1
        return raw

    return ASR(runner, binding, ("<pad>", "a", "b")), calls


class PredictDetailsTests(unittest.TestCase):
    def test_default_predict_returns_text_with_single_runner_call(self):
        task, calls = make_task()
        # Fixture ids [1,1,0,1]: CTC keeps both separated repeats.
        self.assertEqual(task.predict(np.ones(3, np.float32), 16000), "aa")
        self.assertEqual(len(calls), 1)

    def test_return_details_matches_explicit_stages(self):
        task, calls = make_task()
        wave = np.array([0.5, -0.5, 1.0], np.float32)
        details = task.predict(wave, 16000, return_details=True)
        manual = task.preprocess(wave, 16000)
        manual_text = task.postprocess(
            task.infer({task.binding.input_name: manual.tensor})
        )
        self.assertEqual(details.text, manual_text)
        self.assertEqual(details.text, "aa")
        self.assertEqual(details.prepared.valid_samples, manual.valid_samples)
        self.assertEqual(details.prepared.source_rate, manual.source_rate)
        np.testing.assert_array_equal(details.prepared.tensor, manual.tensor)
        # One production execution per request: two requests, two calls.
        self.assertEqual(len(calls), 2)

    def test_details_are_per_call_and_leave_no_model_state(self):
        task, calls = make_task()
        stereo = np.stack(
            [np.linspace(-1, 1, 800, dtype=np.float32), np.zeros(800, np.float32)],
            axis=1,
        )
        first = task.predict(stereo, 8000, return_details=True)
        second = task.predict(np.array([1.0, 2.0], np.float32), 16000, return_details=True)
        self.assertEqual(first.prepared.valid_samples, 1600)
        self.assertEqual(first.prepared.source_rate, 8000)
        self.assertEqual(second.prepared.valid_samples, 2)
        self.assertEqual(second.prepared.source_rate, 16000)
        self.assertEqual(len(calls), 2)
        self.assertEqual(
            set(vars(task)), {"runner", "binding", "vocabulary", "config", "decode_mode"}
        )

    def test_details_record_is_local_and_owned(self):
        from samples.speech.asr.runtime.python.asr import ChunkPrediction

        task, _ = make_task()
        details = task.predict(np.ones(2, np.float32), 16000, return_details=True)
        self.assertIsInstance(details, ChunkPrediction)
        tensor_before = details.prepared.tensor.copy()
        details.prepared.tensor[:] = 0
        again = task.predict(np.ones(2, np.float32), 16000, return_details=True)
        np.testing.assert_array_equal(again.prepared.tensor, tensor_before)


class EntryLoopTests(unittest.TestCase):
    def test_main_runs_one_predict_per_chunk_with_details(self):
        from samples.speech.asr.runtime.python import audio_io, model_runner
        from samples.speech.asr.runtime.python.model_binding import resolve_selection

        original = asr_module.ASR.predict
        seen = []

        def counting(self, waveform, sample_rate, **kwargs):
            seen.append((waveform.shape, sample_rate, kwargs))
            return original(self, waveform, sample_rate, **kwargs)

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            model = root / "model.hbm"
            model.write_bytes(b"test model")
            selection = resolve_selection(
                "s100", asset_id="s:asr:s100/asr.hbm", model_path=model
            )
            runner = model_runner.RuntimeModelRunner(
                selection, runtime=FakeRuntime()
            )

            def chunks(*args):
                yield audio_io.AudioChunk(
                    np.array([0.1, 0.2], np.float32), 16000, 0, 0
                )
                yield audio_io.AudioChunk(
                    np.array([0.3], np.float32), 16000, 1, 1
                )

            args = [
                "--target", "s100",
                "--asset-id", selection.asset.reference,
                "--model-path", str(model),
                "--output-dir", str(root / "out"),
            ]
            with patch(
                "samples._shared.platforms.require_execution_target",
                return_value="s100",
            ), patch.object(
                model_runner, "RuntimeModelRunner", return_value=runner
            ), patch.object(
                audio_io, "read_chunks", side_effect=chunks
            ), patch.object(
                asr_module.ASR, "predict", counting
            ), contextlib.redirect_stdout(
                io.StringIO()
            ), contextlib.redirect_stderr(
                io.StringIO()
            ):
                from samples.speech.asr.runtime.python.main import main

                self.assertEqual(main(args), 0)
            self.assertEqual(len(seen), 2)
            self.assertTrue(all(entry[2] == {"return_details": True} for entry in seen))
            self.assertEqual(
                [entry[0] for entry in seen], [(2,), (1,)]
            )
            report = json.loads((root / "out" / "result.json").read_text())
            self.assertEqual(
                [c["valid_target_samples"] for c in report["chunks"]], [2, 1]
            )
            self.assertEqual(
                [c["source_frames"] for c in report["chunks"]], [2, 1]
            )
            self.assertEqual(report["status"], "completed")


if __name__ == "__main__":
    unittest.main()
