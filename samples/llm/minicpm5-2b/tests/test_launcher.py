"""Host orchestration checks; no models, SDK, network, or board."""

import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

RUNTIME = Path(__file__).resolve().parents[1] / "runtime"
spec = importlib.util.spec_from_file_location(
    "minicpm_launcher", RUNTIME / "launcher.py"
)
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


class LauncherTests(unittest.TestCase):
    def call(self, args):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            rc = launcher.main(args)
        return rc, out.getvalue(), err.getvalue()

    def test_preview_routes_targets_without_execution(self):
        for target, backend in [
            ("s100", "legacy"),
            ("s100p", "legacy"),
            ("s600", "cpp"),
        ]:
            with self.subTest(target=target), patch.object(
                launcher.subprocess, "run"
            ) as run:
                rc, out, err = self.call(
                    ["--target", target, "--dry-run", "--", "--prompt", "two words"]
                )
                self.assertEqual(rc, 0, err)
                plan = json.loads(out)
                self.assertEqual(plan["backend"], backend)
                self.assertEqual(plan["argv"][-2:], ["--prompt", "two words"])
                self.assertFalse(plan["executed"])
                run.assert_not_called()

    def test_board_gate_precedes_build_or_run(self):
        for extra in ([], ["--build"]):
            with patch.object(
                launcher,
                "require_execution_target",
                side_effect=ValueError("board mismatch"),
            ), patch.object(launcher.subprocess, "run") as run:
                rc, _, err = self.call(["--target", "s600", *extra])
                self.assertEqual(rc, 2)
                self.assertIn("board mismatch", err)
                run.assert_not_called()

    def test_build_preview_is_only_cmake(self):
        rc, out, err = self.call(
            ["--target", "s600", "--runtime-root", "/sdk", "--build", "--dry-run"]
        )
        self.assertEqual(rc, 0, err)
        plan = json.loads(out)
        self.assertEqual(len(plan["commands"]), 2)
        self.assertEqual(plan["commands"][0][0], "cmake")
        self.assertIn("-DMINICPM_TARGET=s600", plan["commands"][0])

    def test_missing_binary_never_builds(self):
        with tempfile.TemporaryDirectory() as d, patch.object(
            launcher, "require_execution_target"
        ), patch.object(launcher.subprocess, "run") as run:
            rc, _, err = self.call(
                ["--target", "s600", "--build-dir", d, "--runtime-root", d]
            )
            self.assertEqual(rc, 2)
            run.assert_not_called()

    def test_legacy_timeout_and_target_filename(self):
        rc, out, err = self.call(
            ["--target", "s100p", "--model-dir", "/models/s100p", "--dry-run"]
        )
        self.assertEqual(rc, 0, err)
        plan = json.loads(out)
        self.assertEqual(plan["timeout_seconds"], 120)
        self.assertIn("/models/s100p/minicpm5-2b_ctx4096_s100p.hbm", plan["argv"])

    def test_native_arguments_environment_and_exit_code(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "include/oellm_runtime_basic").mkdir(parents=True)
            (root / "include/oellm_runtime_basic/oellm_runtime.h").write_text("fixture")
            binary = root / "main"
            binary.write_text(
                '#!/bin/sh\n[ "$HB_DNN_USER_DEFINED_L2M_SIZES" = 6:6:6:6 ] || exit 91\n[ "$2" = --prompt ] || exit 92\n[ "$3" = "two words" ] || exit 93\nexit 7\n'
            )
            binary.chmod(0o755)
            with patch.object(launcher, "require_execution_target"):
                rc, _, err = self.call(
                    [
                        "--target",
                        "s600",
                        "--runtime-root",
                        d,
                        "--build-dir",
                        d,
                        "--",
                        "--prompt",
                        "two words",
                    ]
                )
            self.assertEqual(rc, 7, err)

    def test_real_host_rejected(self):
        # The fixture host has no board; this exercises the real shared identity gate.
        with patch.object(launcher.subprocess, "run") as run:
            rc, _, err = self.call(["--target", "s600"])
            self.assertEqual(rc, 2, err)
            run.assert_not_called()

    def test_reject_unsupported_target_and_build_arguments(self):
        for args in (
            ["--target", "x5", "--dry-run"],
            ["--target", "s600", "--build", "--dry-run", "--", "--prompt", "hello"],
        ):
            self.assertEqual(self.call(args)[0], 2)


if __name__ == "__main__":
    unittest.main()
