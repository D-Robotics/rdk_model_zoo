import contextlib
import io
import json
import unittest
from unittest.mock import patch
from samples.speech.asr.runtime.cpp import launcher


class LauncherTests(unittest.TestCase):
    def test_preparation_and_unsupported_targets(self):
        for target in ("s100", "s600"):
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertEqual(launcher.main(["--target", target, "--dry-run"]), 0)
            record = json.loads(output.getvalue())
            self.assertEqual(record["asset_id"], f"s:asr:{target}/asr.hbm")
            self.assertFalse(record["executed"])
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(launcher.main(["--target", "s100p", "--dry-run"]), 2)

    def test_identity_gate_before_build_or_process(self):
        with patch.object(
            launcher,
            "require_execution_target",
            side_effect=ValueError("Target mismatch"),
        ), patch.object(launcher.subprocess, "run") as run, contextlib.redirect_stderr(
            io.StringIO()
        ):
            self.assertEqual(launcher.main(["--target", "s100", "--build"]), 2)
        run.assert_not_called()

    def test_result_rejects_fixture_and_wrong_digest(self):
        record = {
            "target": "s100",
            "asset_id": "s:asr:s100/asr.hbm",
            "model_sha256": "a" * 64,
            "audio_sha256": "b" * 64,
            "decode_mode": "ctc",
        }
        with self.assertRaises(ValueError):
            launcher.validate_report({"execution_backend": "host-fixture"}, record)

    def test_nonzero_native_process_is_recorded_as_executed(self):
        import subprocess
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as temp:
            record = {"executed": False, "processes": []}
            with patch.object(
                launcher.subprocess,
                "run",
                return_value=subprocess.CompletedProcess(
                    ["fixture"], 2, b"partial\xff", b"failure"
                ),
            ), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
                io.StringIO()
            ):
                with self.assertRaises(RuntimeError):
                    launcher.run_logged(["fixture"], "native", Path(temp), record)
            self.assertTrue(record["executed"])
            self.assertEqual(
                (Path(temp) / "native.stdout.log").read_bytes(), b"partial\xff"
            )

    def test_valid_report_and_corrupted_metadata_or_chunks(self):
        import copy

        record = {
            "target": "s100",
            "asset_id": "s:asr:s100/asr.hbm",
            "model_sha256": "a" * 64,
            "audio_sha256": "b" * 64,
            "decode_mode": "ctc",
        }
        report = dict(
            record,
            schema="rdk-model-zoo/asr-native-run/v1",
            status="completed",
            execution_backend="native-sdk",
            vocabulary_sha256=launcher.SHA256,
            config={"audio_maxlen": 30000, "new_rate": 16000},
            metadata={
                "model_name": "unit-fixture",
                "input_shape": [1, 30000],
                "output_shape": [1, 4, 3503],
                "dtype": "float32",
                "input_strides": [120000, 4],
                "output_strides": [56048, 14012, 4],
                "input_bytes": 120000,
                "output_bytes": 56048,
            },
            chunks=[
                {
                    "index": 0,
                    "source_start": 0,
                    "source_frames": 30000,
                    "source_rate": 16000,
                    "valid_target_samples": 30000,
                    "text": "AA",
                }
            ],
            text="AA",
        )
        launcher.validate_report(report, record)
        mutations = [
            lambda r: r.update(text="wrong"),
            lambda r: r["chunks"][0].update(source_start=1),
            lambda r: r["metadata"].update(output_bytes=4),
            lambda r: r.update(model_sha256="0" * 64),
            lambda r: r.update(execution_backend="host-fixture"),
            lambda r: r["metadata"].update(input_shape=[True, 30000]),
            lambda r: r["chunks"][0].update(source_frames=1),
        ]
        for mutate in mutations:
            broken = copy.deepcopy(report)
            mutate(broken)
            with self.assertRaises(ValueError):
                launcher.validate_report(broken, record)

    def test_launcher_success_and_process_failure_keep_records(self):
        import hashlib
        import subprocess
        import tempfile
        from pathlib import Path

        for failure in (False, True):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                model = root / "model"
                model.write_bytes(b"unit fixture model")
                binary = root / "binary"
                binary.write_text("#!/bin/sh\nexit 0\n")
                binary.chmod(0o700)
                output = root / "output"

                def process(argv, **kwargs):
                    if failure:
                        return subprocess.CompletedProcess(
                            argv, 2, b"partial", b"failure"
                        )
                    values = dict(zip(argv[1::2], argv[2::2]))
                    destination = Path(values["--output-dir"])
                    destination.mkdir()
                    report = {
                        "schema": "rdk-model-zoo/asr-native-run/v1",
                        "status": "completed",
                        "execution_backend": "native-sdk",
                        "target": "s100",
                        "asset_id": "s:asr:s100/asr.hbm",
                        "model_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
                        "audio_sha256": launcher.sha256_file(
                            launcher.SAMPLE_DIR / "test_data/chi_sound.wav"
                        ),
                        "vocabulary_sha256": launcher.SHA256,
                        "decode_mode": "ctc",
                        "config": {"audio_maxlen": 30000, "new_rate": 16000},
                        "metadata": {
                            "model_name": "unit-fixture",
                            "input_shape": [1, 30000],
                            "output_shape": [1, 4, 3503],
                            "dtype": "float32",
                            "input_strides": [120000, 4],
                            "output_strides": [56048, 14012, 4],
                            "input_bytes": 120000,
                            "output_bytes": 56048,
                        },
                        "chunks": [
                            {
                                "index": 0,
                                "source_start": 0,
                                "source_frames": 30000,
                                "source_rate": 16000,
                                "valid_target_samples": 30000,
                                "text": "AA",
                            }
                        ],
                        "text": "AA",
                    }
                    (destination / "result.json").write_text(json.dumps(report))
                    return subprocess.CompletedProcess(argv, 0, b"fixture", b"")

                with patch.object(
                    launcher, "require_execution_target", return_value="s100"
                ), patch.object(
                    launcher.subprocess, "run", side_effect=process
                ), contextlib.redirect_stdout(
                    io.StringIO()
                ), contextlib.redirect_stderr(
                    io.StringIO()
                ):
                    rc = launcher.main(
                        [
                            "--target",
                            "s100",
                            "--asset-id",
                            "s:asr:s100/asr.hbm",
                            "--model-path",
                            str(model),
                            "--binary",
                            str(binary),
                            "--output-dir",
                            str(output),
                        ]
                    )
                self.assertEqual(rc, 2 if failure else 0)
                record = json.loads((output / "launch-report.json").read_text())
                self.assertEqual(record["status"], "failed" if failure else "completed")
                self.assertTrue(record["executed"])
                self.assertEqual(
                    record["processes"][0]["returncode"], 2 if failure else 0
                )
                self.assertEqual(record["runtime_metadata_verified"], not failure)
