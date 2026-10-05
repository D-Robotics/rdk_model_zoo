"""Entry readability: main drives pipeline.predict through application helpers.

``main`` keeps the visible per-utterance loop — frontend preparation, one
``bundle.pipeline.predict(features, valid_frames)`` per utterance, evidence
records — while ``application`` exposes the same helpers its compatibility
``run``/``execute`` compositions use, so there is one implementation of the
evidence discipline and no second inference chain.
"""

import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import numpy as np

from samples.speech.paraformer.runtime.python import application, input_io
from samples.speech.paraformer.runtime.python.application import Preparation

TOKENS = (
    Path(__file__).resolve().parents[4]
    / "docs/releases/unified-migration/evidence/2026-09-28-b10-paraformer-pipeline/published-tokens.json"
)


def fake_frontend(valid_frames=7):
    return SimpleNamespace(
        pre_process=lambda audio, rate: SimpleNamespace(
            tensor=np.zeros((1, 400, 560), np.float32),
            valid_frames=valid_frames,
            original_frames=valid_frames + 3,
            truncated=True,
            sample_count=len(audio),
        )
    )


def fake_prediction(text="中"):
    return SimpleNamespace(
        text=text,
        token_ids=(3, 4),
        token_count=2,
        timings_ms={"encoder": 1.0, "predictor": 1.0, "cif": 1.0, "decoder": 1.0},
        decoder_executed=True,
    )


def inference_args(root, audio):
    return [
        "--target", "s100",
        "--audio-file", str(audio),
        "--output-dir", str(root / "run"),
        "--tokens-path", str(TOKENS),
    ]


def add_model_paths(args, root):
    from samples.speech.paraformer.runtime.python.model_binding import (
        resolve_selections,
    )

    for selected in resolve_selections("s100"):
        path = root / f"{selected.stage}.hbm"
        path.write_bytes(b"host-only SDK double; not a compiled HBM")
        args += [
            f"--{selected.stage}-model-path", str(path),
            f"--{selected.stage}-asset-id", selected.asset.reference,
        ]
    return args


class MainLoopTests(unittest.TestCase):
    def test_main_drives_pipeline_predict_without_application_run(self):
        name = "samples.speech.paraformer.runtime.python.main"
        self.assertIsNotNone(importlib.util.find_spec(name))
        main = importlib.import_module(name)
        predict = Mock(return_value=fake_prediction())
        bundle = SimpleNamespace(
            runners=(SimpleNamespace(binding=SimpleNamespace(metadata={})),),
            pipeline=SimpleNamespace(predict=predict),
            set_scheduling_params=lambda **kwargs: None,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio = root / "one.wav"
            audio.write_bytes(b"fixture audio read by injected loader")
            args = add_model_paths(inference_args(root, audio), root)
            stream = io.StringIO()
            with patch(
                "samples._shared.platforms.require_execution_target",
                return_value=None,
            ), patch(
                "samples.speech.paraformer.runtime.python.frontend.ParaformerFrontend",
                return_value=fake_frontend(),
            ), patch(
                "samples.speech.paraformer.runtime.python.runtime.load_runtime",
                return_value=bundle,
            ), patch(
                "samples.speech.paraformer.runtime.python.input_io.read_audio",
                return_value=(np.zeros(16000, np.float32), 16000),
            ), patch.object(
                application, "run", MagicMock()
            ) as run_mock, contextlib.redirect_stdout(stream):
                self.assertEqual(main.main(args), 0)
            run_mock.assert_not_called()
            predict.assert_called_once()
            self.assertEqual(predict.call_args.args[0].shape, (1, 400, 560))
            self.assertEqual(predict.call_args.args[1], 7)
            report = json.loads((root / "run" / "result.json").read_text())
            self.assertEqual(report["utterances"][0]["text"], "中")
            self.assertEqual(report["utterances"][0]["token_count"], 2)
            self.assertTrue(report["inference_executed"])
            self.assertTrue(report["inference_attempted"])
            self.assertEqual(len(report["metadata"]), 1)

    def test_failed_pipeline_call_keeps_null_execution_flag(self):
        main = importlib.import_module(
            "samples.speech.paraformer.runtime.python.main"
        )
        predict = Mock(side_effect=RuntimeError("fixture pipeline failure"))
        bundle = SimpleNamespace(
            runners=(SimpleNamespace(binding=SimpleNamespace(metadata={})),),
            pipeline=SimpleNamespace(predict=predict),
            set_scheduling_params=lambda **kwargs: None,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio = root / "one.wav"
            audio.write_bytes(b"fixture audio read by injected loader")
            args = add_model_paths(inference_args(root, audio), root)
            with patch(
                "samples._shared.platforms.require_execution_target",
                return_value=None,
            ), patch(
                "samples.speech.paraformer.runtime.python.frontend.ParaformerFrontend",
                return_value=fake_frontend(),
            ), patch(
                "samples.speech.paraformer.runtime.python.runtime.load_runtime",
                return_value=bundle,
            ), patch(
                "samples.speech.paraformer.runtime.python.input_io.read_audio",
                return_value=(np.zeros(16000, np.float32), 16000),
            ), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
                io.StringIO()
            ) as errors:
                self.assertEqual(main.main(args), 2)
            self.assertIn("fixture pipeline failure", errors.getvalue())
            failed = json.loads((root / "run" / "failed.json").read_text())
            self.assertEqual(failed["status"], "failed")
            self.assertTrue(failed["inference_attempted"])
            self.assertIsNone(failed["inference_executed"])


class HelperTests(unittest.TestCase):
    def make_preparation(self, root, preprocess_only=False):
        audio = root / "one.wav"
        audio.write_bytes(b"fixture audio")
        entry = {"utt_id": "one", "text": "reference", "speaker": "test"}
        args = SimpleNamespace(
            output_dir=root / "out", preprocess_only=preprocess_only
        )
        preparation = Preparation(
            items=(input_io.InputItem(entry, audio),),
            vocabulary=None,
            initial={},
            report={
                "status": "running",
                "inference_executed": False,
                "inference_attempted": False,
                "utterances": [],
            },
        )
        return args, preparation, audio

    def test_note_runtime_records_bound_metadata(self):
        report = {}
        application.note_runtime(report, None)
        self.assertNotIn("metadata", report)
        bundle = SimpleNamespace(
            runners=(
                SimpleNamespace(binding=SimpleNamespace(metadata={"model": "a"})),
                SimpleNamespace(binding=SimpleNamespace(metadata={"model": "b"})),
            )
        )
        application.note_runtime(report, bundle)
        self.assertEqual(len(report["metadata"]), 2)

    def test_prepare_utterance_collects_evidence_and_features(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args, preparation, audio = self.make_preparation(root)
            with patch.object(
                input_io,
                "read_audio",
                return_value=(np.zeros(100, np.float32), 16000),
            ):
                utterance = application.prepare_utterance(
                    args, preparation, fake_frontend(), preparation.items[0]
                )
            self.assertEqual(utterance.key, "one")
            self.assertEqual(
                preparation.report["current_utterance"], "one"
            )
            self.assertIn(audio, preparation.initial)
            record = utterance.record
            self.assertEqual(record["utt_id"], "one")
            self.assertEqual(record["sample_rate"], 16000)
            self.assertEqual(record["sample_count"], 100)
            self.assertEqual(record["valid_frames"], 7)
            self.assertEqual(record["original_frames"], 10)
            self.assertTrue(record["truncated"])
            self.assertGreaterEqual(record["frontend_ms"], 0.0)
            self.assertEqual(record["reference_text"], "reference")
            self.assertEqual(utterance.features.tensor.shape, (1, 400, 560))

    def test_mark_attempted_flag_transitions(self):
        report = {"inference_attempted": False, "inference_executed": False}
        application.mark_attempted(report)
        self.assertTrue(report["inference_attempted"])
        self.assertIsNone(report["inference_executed"])
        application.mark_attempted(report)
        self.assertIsNone(report["inference_executed"])
        report["inference_executed"] = True
        application.mark_attempted(report)
        self.assertTrue(report["inference_executed"])

    def test_record_prediction_updates_record_and_report(self):
        report = {"inference_attempted": True, "inference_executed": None,
                  "utterances": []}
        utterance = application.Utterance(
            "one",
            {"utt_id": "one"},
            {"utt_id": "one"},
            SimpleNamespace(tensor=None, valid_frames=7),
            1.5,
        )
        application.record_prediction(report, utterance, fake_prediction())
        self.assertTrue(report["inference_executed"])
        entry = report["utterances"][0]
        self.assertEqual(entry["text"], "中")
        self.assertEqual(entry["token_ids"], [3, 4])
        self.assertEqual(entry["token_count"], 2)
        self.assertTrue(entry["decoder_executed"])
        self.assertIn("timings_ms", entry)

    def test_save_features_writes_npy_manifest_entry_and_record(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args, preparation, _ = self.make_preparation(root, preprocess_only=True)
            (root / "out").mkdir()
            (root / "out" / "feats").mkdir()
            utterance = application.Utterance(
                "one",
                {"utt_id": "one", "text": "reference", "speaker": "test"},
                {"utt_id": "one", "text": "reference"},
                SimpleNamespace(
                    tensor=np.ones((1, 400, 560), np.float32) * 2,
                    valid_frames=7,
                    original_frames=10,
                    truncated=True,
                ),
                0.5,
            )
            prepared_manifest = []
            application.save_features(
                args, preparation, utterance, prepared_manifest
            )
            saved = np.load(root / "out" / "feats" / "one.npy")
            np.testing.assert_array_equal(saved, utterance.features.tensor)
            entry = preparation.report["utterances"][0]
            self.assertEqual(entry["feature_file"], "feats/one.npy")
            self.assertIn("feature_sha256", entry)
            self.assertEqual(len(prepared_manifest), 1)
            self.assertEqual(prepared_manifest[0]["speaker"], "test")
            self.assertEqual(prepared_manifest[0]["feat_length"], 7)
            self.assertEqual(prepared_manifest[0]["original_frames"], 10)
            self.assertTrue(prepared_manifest[0]["truncated"])

    def test_complete_verifies_digests_and_writes_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args, preparation, audio = self.make_preparation(root, preprocess_only=True)
            (root / "out").mkdir()
            preparation.report["current_utterance"] = "one"
            report = application.complete(args, preparation, [])
            self.assertEqual(report["status"], "completed")
            self.assertNotIn("current_utterance", report)
            self.assertTrue((root / "out" / "result.json").is_file())
            self.assertTrue((root / "out" / "prepared-manifest.json").is_file())
            # A digest drift after preparation fails instead of completing.
            preparation2 = Preparation(
                items=preparation.items,
                vocabulary=None,
                initial={audio: "0" * 64},
                report={"status": "running", "utterances": []},
            )
            with self.assertRaisesRegex(ValueError, "changed during execution"):
                application.complete(args, preparation2, [])

    def test_run_composition_still_works_as_compatibility(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args, preparation, _ = self.make_preparation(root)
            (root / "out").mkdir()
            bundle = SimpleNamespace(
                runners=(SimpleNamespace(binding=SimpleNamespace(metadata={})),),
                pipeline=SimpleNamespace(predict=Mock(return_value=fake_prediction())),
            )
            with patch.object(
                input_io,
                "read_audio",
                return_value=(np.zeros(100, np.float32), 16000),
            ):
                report = application.run(
                    args, None, preparation, fake_frontend(), bundle
                )
            self.assertEqual(report["status"], "completed")
            bundle.pipeline.predict.assert_called_once()
            self.assertTrue((root / "out" / "result.json").is_file())


if __name__ == "__main__":
    unittest.main()
