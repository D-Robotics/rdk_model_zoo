import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from samples.speech.asr.runtime.python.model_binding import (
    resolve_selection,
    SAMPLE_DIR,
)
from samples.speech.asr.runtime.python.asr import RuntimeModelRunner
from samples.speech.asr.runtime.python.audio_io import AudioChunk
from samples.speech.asr.runtime.python.main import main


class FakeRuntime:
    model_names = ["asr"]
    input_names = {"asr": ["audio"]}
    output_names = {"asr": ["logits"]}
    input_shapes = {"asr": {"audio": [1, 30000]}}
    output_shapes = {"asr": {"logits": [1, 4, 3503]}}
    input_dtypes = {"asr": {"audio": "float32"}}
    output_dtypes = {"asr": {"logits": "float32"}}
    output_quants = {"asr": {}}

    def __init__(self):
        self.calls = 0
        self.raw = np.zeros((1, 4, 3503), np.float32)
        self.raw[0, :2, 5] = 1
        self.raw[0, 3, 5] = 1

    def run(self, tensors):
        self.calls += 1
        return {"asr": {"logits": self.raw}}

    def set_scheduling_params(self, **kwargs):
        self.scheduling = kwargs


class RuntimeTests(unittest.TestCase):
    def test_help_selection_and_invalid_target_without_sdk(self):
        for target in ("s100", "s600"):
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                self.assertEqual(main(["--target", target, "--dry-run"]), 0)
            data = json.loads(out.getvalue())
            self.assertFalse(data["sdk_loaded"])
            self.assertEqual(data["decode_mode"], "legacy")
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(main(["--target", "s100p", "--dry-run"]), 2)

    def test_sdk_gate_and_output_ownership(self):
        with patch(
            "samples.speech.asr.runtime.python.asr.require_execution_target",
            side_effect=ValueError("Target mismatch"),
        ), patch(
            "utils.py_utils.single_array_runner._default_runtime_factory"
        ) as factory:
            with self.assertRaises(ValueError):
                RuntimeModelRunner(resolve_selection("s100")).load()
        factory.assert_not_called()
        runtime = FakeRuntime()
        runner = RuntimeModelRunner(resolve_selection("s100"), runtime=runtime)
        result = runner({"audio": np.zeros((1, 30000), np.float32)})
        runtime.raw[:] = 0
        self.assertEqual(float(result.max()), 1)

    def test_success_and_partial_failure_reports(self):
        from samples.speech.asr.runtime.python import asr as asr_module, audio_io

        for fail in (False, True):
            with self.subTest(fail=fail), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                model = root / "model.hbm"
                model.write_bytes(b"test model")
                selection = resolve_selection(
                    "s100", asset_id="s:asr:s100/asr.hbm", model_path=model
                )
                runner = RuntimeModelRunner(selection, runtime=FakeRuntime())

                def chunks(*args):
                    yield AudioChunk(np.array([0.1, 0.2], np.float32), 16000, 0, 0)
                    if fail:
                        raise ValueError("fixture late read failure")
                    yield AudioChunk(np.array([0.1, 0.2], np.float32), 16000, 2, 1)

                args = [
                    "--target",
                    "s100",
                    "--asset-id",
                    selection.asset.reference,
                    "--model-path",
                    str(model),
                    "--output-dir",
                    str(root / "out"),
                ]
                with patch(
                    "utils.py_utils.platforms.require_execution_target",
                    return_value="s100",
                ), patch.object(
                    asr_module, "RuntimeModelRunner", return_value=runner
                ), patch.object(
                    audio_io, "read_chunks", side_effect=chunks
                ), contextlib.redirect_stdout(
                    io.StringIO()
                ), contextlib.redirect_stderr(
                    io.StringIO()
                ):
                    self.assertEqual(main(args), 2 if fail else 0)
                path = root / "out" / ("failed.json" if fail else "result.json")
                record = json.loads(path.read_text())
                self.assertEqual(record["status"], "failed" if fail else "completed")
                self.assertEqual(len(record["chunks"]), 1 if fail else 2)
                if fail:
                    self.assertFalse((root / "out/result.json").exists())
                else:
                    self.assertEqual(record["text"], "AAAAAA")

    def test_report_write_failure_returns_original_error_without_traceback(self):
        from samples.speech.asr.runtime.python import asr as asr_module

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            model = root / "model.hbm"
            model.write_bytes(b"fixture")
            selection = resolve_selection(
                "s100", asset_id="s:asr:s100/asr.hbm", model_path=model
            )
            runner = RuntimeModelRunner(selection, runtime=FakeRuntime())
            original_write = Path.write_text

            def fail_reports(path, *args, **kwargs):
                if path.name in ("result.json", "failed.json"):
                    raise OSError("fixture disk full")
                return original_write(path, *args, **kwargs)

            errors = io.StringIO()
            with patch(
                "utils.py_utils.platforms.require_execution_target",
                return_value="s100",
            ), patch.object(
                asr_module, "RuntimeModelRunner", return_value=runner
            ), patch.object(
                Path, "write_text", fail_reports
            ), contextlib.redirect_stderr(
                errors
            ):
                self.assertEqual(
                    main(
                        [
                            "--target",
                            "s100",
                            "--asset-id",
                            selection.asset.reference,
                            "--model-path",
                            str(model),
                            "--output-dir",
                            str(root / "out"),
                        ]
                    ),
                    2,
                )
            self.assertIn("fixture disk full", errors.getvalue())
            self.assertIn("Could not save failure report", errors.getvalue())
