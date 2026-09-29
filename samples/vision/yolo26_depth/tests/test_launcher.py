"""Native launcher orchestration fixtures; no board or SDK execution."""

import contextlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
from samples.vision.yolo26_depth.runtime.cpp import launcher
from samples.vision.yolo26_depth.runtime.python.model_binding import resolve_selection


class LauncherTests(unittest.TestCase):
    def test_dry_run_and_unsupported_target_without_build(self):
        with patch.object(launcher.subprocess, "run") as run:
            with contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(
                    launcher.main(
                        ["--target", "x5", "--variant", "l", "--dry-run", "--build"]
                    ),
                    0,
                )
            self.assertEqual(json.loads(output.getvalue())["target"], "x5")
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(launcher.main(["--target", "s100", "--dry-run"]), 2)
            run.assert_not_called()

    def test_identity_gate_before_build(self):
        with patch.object(
            launcher, "require_execution_target", side_effect=ValueError("wrong board")
        ) as gate, patch.object(
            launcher.subprocess, "run"
        ) as run, contextlib.redirect_stderr(
            io.StringIO()
        ):
            self.assertEqual(launcher.main(["--target", "x5", "--build"]), 2)
            gate.assert_called_once_with("x5")
            run.assert_not_called()

    def test_custom_native_success_binds_exact_files_and_retains_logs(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            model = p / "model.bin"
            model.write_bytes(b"custom")
            image = p / "image.png"
            image.write_bytes(b"fixture input")
            binary = p / "binary"
            binary.write_bytes(b"fixture binary")
            identity = resolve_selection("x5").asset.reference

            def run(command, **kwargs):
                out = Path(command[command.index("--output") + 1])
                out.mkdir()
                (out / "report.json").write_text(
                    json.dumps({"target": "x5", "runtime_version": "fixture"})
                )
                return SimpleNamespace(
                    returncode=0, stdout="native stdout\n", stderr="native stderr\n"
                )

            with patch.object(launcher, "require_execution_target"), patch.object(
                launcher.subprocess, "run", side_effect=run
            ), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
                io.StringIO()
            ):
                self.assertEqual(
                    launcher.main(
                        [
                            "--asset-id",
                            identity,
                            "--model-path",
                            str(model),
                            "--converted-model",
                            "--binary",
                            str(binary),
                            "--test-img",
                            str(image),
                            "--output",
                            str(p / "out"),
                        ]
                    ),
                    0,
                )
            report = json.loads((p / "out/launch-report.json").read_text())
            self.assertIsNone(report["asset_id"])
            self.assertIsNone(report["publisher_sha256"])
            self.assertEqual(report["artifact_origin"], "user-converted")
            for key in (
                "model_sha256",
                "input_sha256",
                "binary_sha256",
                "native_report_sha256",
            ):
                self.assertEqual(len(report[key]), 64)
            self.assertEqual(
                (p / "out/native.stdout.log").read_text(), "native stdout\n"
            )


if __name__ == "__main__":
    unittest.main()
