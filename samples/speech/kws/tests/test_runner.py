import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from samples.speech.kws.runtime.python.model_binding import resolve_selection
from samples.speech.kws.runtime.python.kws import RuntimeModelRunner
from samples.speech.kws.runtime.python.main import main


class FakeRuntime:
    model_names = ["kws"]
    input_names = {"kws": ["features"]}
    output_names = {"kws": ["score"]}
    input_shapes = {"kws": {"features": [1, 373, 80]}}
    output_shapes = {"kws": {"score": [1, 2, 1]}}
    input_dtypes = {"kws": {"features": "float32"}}
    output_dtypes = {"kws": {"score": "float32"}}
    output_quants = {"kws": {}}

    def __init__(self):
        self.raw = np.array([[[0.2], [0.985]]], np.float32)
        self.calls = []

    def run(self, tensors):
        self.calls.append(tensors)
        return {"kws": {"score": self.raw}}

    def set_scheduling_params(self, **kwargs):
        self.scheduling = kwargs


class RunnerTests(unittest.TestCase):
    def test_raw_output_ownership_and_scheduling(self):
        runtime = FakeRuntime()
        runner = RuntimeModelRunner(resolve_selection("s100"), runtime=runtime)
        runner.set_scheduling_params(priority=3, bpu_cores=[0])
        self.assertEqual(
            runtime.scheduling, {"priority": {"kws": 3}, "bpu_cores": {"kws": [0]}}
        )
        result = runner({"features": np.zeros((1, 373, 80), np.float32)})
        runtime.raw[:] = 0
        self.assertAlmostEqual(float(result.max()), 0.985, places=6)
        self.assertEqual(len(runtime.calls), 1)
        with self.assertRaises(ValueError):
            runner({"features": np.zeros((1, 373, 80), np.float64)})

    def test_board_gate_precedes_sdk_factory(self):
        with patch(
            "samples.speech.kws.runtime.python.kws.require_execution_target",
            side_effect=ValueError("Target mismatch"),
        ), patch(
            "utils.py_utils.single_array_runner._default_runtime_factory"
        ) as factory:
            runner = RuntimeModelRunner(resolve_selection("s100"))
            with self.assertRaises(ValueError):
                runner.load()
        factory.assert_not_called()

    def test_main_end_to_end_report_with_explicit_test_doubles(self):
        from samples.speech.kws.runtime.python import audio_io, frontend, kws as kws_module

        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            model = folder / "kws.hbm"
            model.write_bytes(b"fake model")
            audio = folder / "audio.wav"
            audio.write_bytes(b"fake decoded by test")
            selection = resolve_selection(
                "s100", asset_id="s:kws:s100/kws.hbm", model_path=model
            )
            runner = RuntimeModelRunner(selection, runtime=FakeRuntime())
            args = [
                "--target",
                "s100",
                "--asset-id",
                selection.asset.reference,
                "--model-path",
                str(model),
                "--audio-file",
                str(audio),
                "--output-dir",
                str(folder / "output"),
            ]
            with patch(
                "utils.py_utils.platforms.require_execution_target",
                return_value="s100",
            ), patch.object(
                kws_module, "RuntimeModelRunner", return_value=runner
            ), patch.object(
                audio_io, "load_audio", return_value=(np.ones(40000, np.float32), 16000)
            ), patch.object(
                frontend, "paddle_fbank", return_value=np.zeros((373, 80), np.float32)
            ), contextlib.redirect_stdout(
                io.StringIO()
            ):
                self.assertEqual(main(args), 0)
            report = json.loads((folder / "output/result.json").read_text())
            self.assertEqual(report["padded_samples"], 20000)
            self.assertTrue(report["detected"])
            self.assertIsNone(report["publisher_sha256"])
            with patch(
                "utils.py_utils.platforms.require_execution_target",
                return_value="s100",
            ), patch.object(
                audio_io, "load_audio", return_value=(np.ones(40000, np.float32), 16000)
            ), contextlib.redirect_stderr(
                io.StringIO()
            ):
                self.assertEqual(main(args), 2)
