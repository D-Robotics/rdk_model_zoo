"""Native launch contract using fixtures only; no SDK or board access."""

import contextlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
from samples.vision.lanenet.runtime.cpp import launcher
from samples.vision.lanenet.runtime.python.model_binding import ASSET_ID


class LauncherTests(unittest.TestCase):
    def test_discovery_never_builds_or_checks_board(self):
        with patch.object(launcher.subprocess, "run") as run, patch.object(
            launcher, "require_execution_target"
        ) as gate:
            for mode in ("--dry-run", "--list-models"):
                with contextlib.redirect_stdout(io.StringIO()) as output:
                    self.assertEqual(launcher.main([mode, "--build"]), 0)
                    self.assertTrue(json.loads(output.getvalue()))
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(launcher.main(["--target", "s100p", "--dry-run"]), 2)
            gate.assert_not_called()
            run.assert_not_called()

    def test_identity_gate_precedes_build(self):
        with patch.object(
            launcher, "require_execution_target", side_effect=ValueError("wrong board")
        ) as gate, patch.object(
            launcher.subprocess, "run"
        ) as run, contextlib.redirect_stderr(
            io.StringIO()
        ):
            self.assertEqual(launcher.main(["--build"]), 2)
            gate.assert_called_once_with("s100")
            run.assert_not_called()

    def test_native_run_preserves_logs_and_exact_file_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            for name in ("model.hbm", "input.jpg", "binary"):
                (p / name).write_bytes(name.encode())
            args = [
                "--asset-id",
                ASSET_ID,
                "--model-path",
                str(p / "model.hbm"),
                "--test-img",
                str(p / "input.jpg"),
                "--binary",
                str(p / "binary"),
                "--output",
                str(p / "out"),
            ]

            def execute(command, **kwargs):
                self.assertEqual(kwargs["cwd"], launcher.ROOT)
                out = Path(command[command.index("--output") + 1])
                out.mkdir()
                (out / "report.json").write_text('{"fixture":true}')
                return SimpleNamespace(
                    returncode=0, stdout="full stdout\n", stderr="full stderr\n"
                )

            with patch.object(launcher, "require_execution_target"), patch.object(
                launcher.subprocess, "run", side_effect=execute
            ), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
                io.StringIO()
            ):
                self.assertEqual(launcher.main(args), 0)
            record = json.loads((p / "out/launch-report.json").read_text())
            self.assertTrue(record["executed"])
            self.assertIsNone(record["publisher_sha256"])
            for key in (
                "model_sha256",
                "input_sha256",
                "binary_sha256",
                "native_report_sha256",
            ):
                self.assertEqual(len(record[key]), 64)
            self.assertEqual((p / "out/native.stdout.log").read_text(), "full stdout\n")
            self.assertEqual((p / "out/native.stderr.log").read_text(), "full stderr\n")

    def test_extra_path_conflicts_rejected_before_build(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            for name in ("model.hbm", "input.jpg"):
                (p / name).write_bytes(b"fixture")
            args = [
                "--asset-id",
                ASSET_ID,
                "--model-path",
                str(p / "model.hbm"),
                "--test-img",
                str(p / "input.jpg"),
                "--output",
                str(p / "out"),
                "--build",
            ]
            with patch.object(launcher, "require_execution_target"), patch.object(
                launcher.subprocess, "run"
            ) as run, contextlib.redirect_stderr(io.StringIO()):
                for name in (
                    "launch-report.json",
                    "raw_output_0.npy",
                    "native.stdout.log",
                ):
                    self.assertEqual(
                        launcher.main(
                            args + ["--instance-save-path", str(p / "out" / name)]
                        ),
                        2,
                    )
                self.assertEqual(
                    launcher.main(
                        args
                        + [
                            "--instance-save-path",
                            str(p / "same.png"),
                            "--binary-save-path",
                            str(p / "same.png"),
                        ]
                    ),
                    2,
                )
                run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
