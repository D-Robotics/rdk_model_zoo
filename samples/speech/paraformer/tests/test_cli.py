"""Manifest preservation and explicit host-only feature workflow."""

import contextlib
import importlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np


class CliTests(unittest.TestCase):
    def setUp(self):
        name = "samples.speech.paraformer.runtime.python.main"
        self.assertIsNotNone(importlib.util.find_spec(name), "Missing Paraformer CLI")
        self.cli = importlib.import_module(name)
        self.io = importlib.import_module(
            "samples.speech.paraformer.runtime.python.cli"
        )

    def test_manifest_rejects_duplicate_or_escaping_ids_and_negative_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.json"
            for items in (
                [{"utt_id": "../escape"}],
                [{"utt_id": "a"}, {"utt_id": "a"}],
                [{"utt_id": "a", "text": 5}],
            ):
                manifest.write_text(json.dumps(items))
                with self.assertRaises(ValueError):
                    self.io.load_manifest(manifest, root, 0)
            manifest.write_text('[{"utt_id":"a"}]')
            with self.assertRaises(ValueError):
                self.io.load_manifest(manifest, root, -1)

    def test_missing_selected_wav_fails_instead_of_silent_skip(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.json"
            manifest.write_text('[{"utt_id":"missing"}]')
            with self.assertRaises(ValueError):
                self.io.load_manifest(manifest, root, 0)

    def test_preprocess_writes_separate_manifest_and_refuses_existing_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio = root / "one.wav"
            audio.write_bytes(b"fixture audio read by injected loader")
            manifest = root / "manifest.json"
            original = '[{"utt_id":"one","text":"reference","speaker":"test"}]\n'
            manifest.write_text(original)
            out = root / "prepared"

            class Frontend:
                def __init__(self, *args, **kwargs):
                    pass

                def pre_process(self, samples, rate):
                    return SimpleNamespace(
                        tensor=np.zeros((1, 400, 560), np.float32),
                        valid_frames=400,
                        original_frames=500,
                        truncated=True,
                        sample_count=len(samples),
                    )

            args = [
                "--target",
                "s100",
                "--preprocess-only",
                "--manifest",
                str(manifest),
                "--audio-dir",
                str(root),
                "--output-dir",
                str(out),
            ]
            with patch(
                "samples.speech.paraformer.runtime.python.frontend.ParaformerFrontend",
                Frontend,
            ), patch.object(
                self.io, "read_audio", return_value=(np.zeros(16000, np.float32), 16000)
            ), contextlib.redirect_stdout(
                io.StringIO()
            ), contextlib.redirect_stderr(
                io.StringIO()
            ):
                self.assertEqual(self.cli.main(args), 0)
                self.assertEqual(self.cli.main(args), 2)
            self.assertEqual(manifest.read_text(), original)
            prepared = json.loads((out / "prepared-manifest.json").read_text())
            self.assertEqual(prepared[0]["feat_length"], 400)
            self.assertEqual(prepared[0]["speaker"], "test")
            self.assertTrue(prepared[0]["truncated"])
            self.assertEqual(np.load(out / "feats/one.npy").shape, (1, 400, 560))
            report = json.loads((out / "result.json").read_text())
            self.assertEqual(report["status"], "completed")
            self.assertFalse(report["inference_executed"])
            self.assertNotIn("text", report["utterances"][0])

    def test_dry_run_does_not_load_sdk_frontend_or_write_output(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "new"
            with patch(
                "samples.speech.paraformer.runtime.python.frontend.ParaformerFrontend",
                side_effect=AssertionError("must not load"),
            ), contextlib.redirect_stdout(io.StringIO()) as captured:
                rc = self.cli.main(
                    ["--target", "s100", "--dry-run", "--output-dir", str(out)]
                )
            self.assertEqual(rc, 0)
            self.assertFalse(out.exists())
            self.assertFalse(json.loads(captured.getvalue())["sdk_loaded"])

    def test_inference_target_gate_precedes_frontend_and_output_creation(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "new"
            with patch(
                "utils.py_utils.platforms.require_execution_target",
                side_effect=ValueError("Target mismatch"),
            ), patch(
                "samples.speech.paraformer.runtime.python.frontend.ParaformerFrontend",
                side_effect=AssertionError("must not load"),
            ), contextlib.redirect_stderr(
                io.StringIO()
            ):
                rc = self.cli.main(["--target", "s100", "--output-dir", str(output)])
            self.assertEqual(rc, 2)
            self.assertFalse(output.exists())

    def test_full_cli_uses_real_shared_pipeline_with_explicit_sdk_doubles(self):
        from samples.speech.paraformer.tests.test_binding import metadata
        from samples.speech.paraformer.runtime.python.cli import resolve_selections
        from samples.speech.paraformer.runtime.python import pipeline

        original_loader = pipeline.ParaformerPipeline.from_models
        root_repo = Path(__file__).resolve().parents[4]
        model_calls = []
        loaded_stages = []
        schedules = {}
        model_feeds = {}

        def factory(path):
            stage = Path(path).stem
            meta = metadata(stage)
            sdk = SimpleNamespace(model_names=[stage])
            for field in (
                "input_names",
                "input_shapes",
                "input_dtypes",
                "output_names",
                "output_shapes",
                "output_dtypes",
            ):
                setattr(sdk, field, {stage: getattr(meta, field)})

            def run(inputs):
                model_calls.append(stage)
                model_feeds[stage] = inputs[stage]
                arrays = {
                    n: np.zeros(shape, dtype=meta.output_dtypes[n])
                    for n, shape in meta.output_shapes.items()
                }
                if stage == "predictor":
                    arrays["/predictor/Add_output_0"][0, :2] = 1
                    arrays["/predictor/Concat_5_output_0"][0, :2] = 3
                if stage == "decoder":
                    arrays["logits"][0, :2, 3] = 1
                    arrays["token_num"][:] = 2
                return {stage: arrays}

            loaded_stages.append(stage)
            schedules[stage] = []
            sdk.set_scheduling_params = lambda **kw: schedules[stage].append(kw)
            sdk.run = run
            return sdk

        class Frontend:
            def __init__(self, *args, **kwargs):
                pass

            def pre_process(self, audio, rate):
                return SimpleNamespace(
                    tensor=np.full((1, 400, 560), 0.125, np.float32),
                    valid_frames=2,
                    original_frames=2,
                    truncated=False,
                    sample_count=len(audio),
                )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio = root / "one.wav"
            audio.write_bytes(b"synthetic input for explicitly mocked audio loader")
            out = root / "run"
            tokens = (
                root_repo
                / "samples/speech/paraformer/tests/fixtures/published-tokens.json"
            )
            args = [
                "--target",
                "s100",
                "--priority", "7",
                "--bpu-cores", "0",
                "--audio-file",
                str(audio),
                "--output-dir",
                str(out),
                "--tokens-path",
                str(tokens),
            ]
            for selected in resolve_selections("s100"):
                path = root / f"{selected.stage}.hbm"
                path.write_bytes(b"host-only SDK double; not a compiled HBM")
                args += [
                    f"--{selected.stage}-model-path",
                    str(path),
                    f"--{selected.stage}-asset-id",
                    selected.asset.reference,
                ]
            with patch(
                "utils.py_utils.platforms.require_execution_target", return_value=None
            ), patch(
                "samples.speech.paraformer.runtime.python.frontend.ParaformerFrontend",
                Frontend,
            ), patch.object(
                self.io, "read_audio", return_value=(np.zeros(1000, np.float32), 16000)
            ), patch.object(
                pipeline.ParaformerPipeline,
                "from_models",
                side_effect=lambda selections, vocabulary: original_loader(
                    selections, vocabulary, runtime_factory=factory
                ),
            ), contextlib.redirect_stdout(
                io.StringIO()
            ):
                self.assertEqual(self.cli.main(args), 0)
            report = json.loads((out / "result.json").read_text())
            self.assertEqual(model_calls, ["encoder", "predictor", "decoder"])
            self.assertEqual(loaded_stages, ["encoder", "predictor", "decoder"])
            self.assertEqual(len(report["metadata"]), 3)
            self.assertTrue(np.all(model_feeds["encoder"]["speech"] == 0.125))
            np.testing.assert_array_equal(model_feeds["decoder"]["token_num"], [2])
            np.testing.assert_array_equal(
                model_feeds["decoder"]["onnx::Shape_8609"][0, :2, 0], [3, 3])
            for stage in loaded_stages:
                self.assertEqual(schedules[stage],
                                 [{"priority": {stage: 7}, "bpu_cores": {stage: [0]}}])
            self.assertEqual(report["utterances"][0]["text"], "andand")
            self.assertTrue(report["inference_executed"])
            self.assertTrue(report["inference_attempted"])
            self.assertEqual(len(report["metadata"]), 3)
            self.assertTrue(all(m["observed_sha256"] for m in report["model_set"]))

    def test_failed_preprocessing_keeps_failure_record_and_input_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "one.wav").write_bytes(b"fixture")
            manifest = root / "manifest.json"
            original = '[{"utt_id":"one"}]'
            manifest.write_text(original)
            out = root / "run"
            with patch.object(
                self.io, "read_audio", side_effect=ValueError("bad audio")
            ), patch(
                "samples.speech.paraformer.runtime.python.frontend.ParaformerFrontend",
                return_value=object(),
            ), contextlib.redirect_stderr(
                io.StringIO()
            ):
                rc = self.cli.main(
                    [
                        "--target",
                        "s100",
                        "--preprocess-only",
                        "--manifest",
                        str(manifest),
                        "--audio-dir",
                        str(root),
                        "--output-dir",
                        str(out),
                    ]
                )
            self.assertEqual(rc, 2)
            self.assertEqual(manifest.read_text(), original)
            self.assertFalse((out / "result.json").exists())
            self.assertEqual(
                json.loads((out / "failed.json").read_text())["status"], "failed"
            )


if __name__ == "__main__":
    unittest.main()
