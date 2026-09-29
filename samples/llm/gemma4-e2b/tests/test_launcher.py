"""Host orchestration tests; no SDK, models, network or board inference."""

import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import unittest
import tempfile
from unittest.mock import patch

CPP = Path(__file__).resolve().parents[1] / "runtime/cpp"
SPEC = importlib.util.spec_from_file_location("gemma_launcher", CPP / "launcher.py")
launcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launcher)


class LauncherTest(unittest.TestCase):
    def test_native_process_receives_exact_args_environment_and_exit_code(self):
        with tempfile.TemporaryDirectory() as directory:
            binary = Path(directory) / "gemma4_demo"
            binary.write_text(
                '#!/bin/sh\n[ "$GEMMA4_TARGET" = s600 ] || exit 91\n'
                '[ "$1" = text ] || exit 92\n'
                "[ \"$2\" = '--prompt' ] || exit 93\n"
                "[ \"$3\" = 'two words' ] || exit 94\nexit 7\n"
            )
            binary.chmod(0o755)
            with patch.object(
                launcher, "require_execution_target", return_value="s600"
            ):
                rc, _, _ = self.call(
                    [
                        "--target",
                        "s600",
                        "--build-dir",
                        directory,
                        "demo",
                        "text",
                        "--prompt",
                        "two words",
                    ]
                )
            self.assertEqual(rc, 7)

    def call(self, args):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            rc = launcher.main(args)
        return rc, out.getvalue(), err.getvalue()

    def test_preview_preserves_all_modes_and_native_arguments(self):
        for mode, binary in launcher.APPS.items():
            with self.subTest(mode=mode), patch.object(
                launcher.subprocess, "run"
            ) as run:
                rc, out, _ = self.call(
                    [
                        "--target",
                        "s600",
                        "--dry-run",
                        mode,
                        "--prompt",
                        "text with spaces",
                        "--max_tokens=8",
                    ]
                )
                self.assertEqual(rc, 0)
                record = json.loads(out)
                self.assertEqual(Path(record["native_argv"][0]).name, binary)
                self.assertEqual(
                    record["native_argv"][1:],
                    ["--prompt", "text with spaces", "--max_tokens=8"],
                )
                self.assertFalse(record["executed"])
                run.assert_not_called()

    def test_real_execution_rejects_host_before_build_or_binary(self):
        with patch.object(
            launcher, "require_execution_target", side_effect=ValueError("no board")
        ), patch.object(launcher.subprocess, "run") as run:
            rc, _, err = self.call(["--target", "s600", "--build"])
            self.assertEqual(rc, 2)
            self.assertIn("no board", err)
            run.assert_not_called()

    def test_build_is_explicit_and_target_is_forwarded(self):
        with patch.object(
            launcher, "require_execution_target", return_value="s100p"
        ), patch.object(launcher.subprocess, "run") as run:
            run.return_value.returncode = 0
            rc, _, _ = self.call(["--target", "s100p", "--build"])
            self.assertEqual(rc, 0)
            self.assertEqual(run.call_count, 1)
            args, kwargs = run.call_args
            self.assertEqual(Path(args[0][1]).name, "build.sh")
            self.assertEqual(kwargs["env"]["GEMMA4_TARGET"], "s100p")

    def test_missing_binary_does_not_build_or_download(self):
        with patch.object(
            launcher, "require_execution_target", return_value="s600"
        ), patch.object(launcher.subprocess, "run") as run:
            rc, _, err = self.call(
                ["--target", "s600", "--build-dir", "/nonexistent/gemma-build"]
            )
            self.assertEqual(rc, 2)
            self.assertIn("--build", err)
            run.assert_not_called()

    def test_environment_is_copied_and_s600_source_settings_preserved(self):
        with patch.dict(
            os.environ, {"LD_LIBRARY_PATH": "old", "GEMMA4_USE_DNN_V3": "1"}
        ):
            env = launcher.execution_environment(
                "s600", Path("/models"), Path("/build")
            )
            self.assertNotIn("LD_LIBRARY_PATH", env)
            self.assertNotIn("GEMMA4_USE_DNN_V3", env)
            self.assertEqual(env["HB_DNN_USER_DEFINED_L2M_SIZES"], "6:6:6:6")
            self.assertEqual(os.environ["LD_LIBRARY_PATH"], "old")

    def test_unknown_mode_and_x5_rejected_without_side_effects(self):
        with patch.object(launcher.subprocess, "run") as run:
            self.assertEqual(self.call(["--target", "x5", "--dry-run"])[0], 2)
            self.assertEqual(self.call(["--target", "s600", "unknown"])[0], 2)
            run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
