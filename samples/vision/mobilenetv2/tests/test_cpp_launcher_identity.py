"""Host identity-fixture tests for the C++ launcher gate (B1-R2).

The launcher resolves board identity from sysfs files; its
SOC_NAME_FILE/BOARD_TYPE_FILE overrides let these tests point it at
fixture files, so the S100P rejection matrix is provable on a host with
no board attached. The expected semantics mirror
samples/_shared/platforms.py:match_target — soc_name=s100 subdivides on
board_type (s100p / "rdk s100p" → S100P), while soc_name=s600 ignores
board_type entirely.
"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[4]
RUN_SH = (
    ROOT / "samples" / "vision" / "mobilenetv2" / "runtime" / "cpp" / "run.sh"
)

S100P_ERROR = "detected s100p"
UNKNOWN_ERROR = "unrecognized or unsupported SoC"


class CppLauncherIdentityTests(unittest.TestCase):
    def _run_launcher(self, soc_name, board_type):
        """Run run.sh against fixture identity files.

        The identity gate runs before any build or model work, so every
        host case terminates either at the gate (identity error) or at
        the model-existence check right after it ("Model not found") —
        the latter is the observable proof that the gate accepted the
        identity and that the check precedes cmake.
        """
        with tempfile.TemporaryDirectory() as tmp:
            soc_file = Path(tmp) / "soc_name"
            board_file = Path(tmp) / "board_type"
            if soc_name is not None:
                soc_file.write_text(soc_name, encoding="utf-8")
            if board_type is not None:
                board_file.write_text(board_type, encoding="utf-8")
            env = dict(os.environ)
            env["SOC_NAME_FILE"] = str(soc_file)
            env["BOARD_TYPE_FILE"] = str(board_file)
            # Never-existing model: accepted identities must stop at the
            # model check, not silently proceed into a build.
            env["MODEL_PATH"] = str(Path(tmp) / "no-such-model.hbm")
            return subprocess.run(
                ["bash", str(RUN_SH)],
                env=env,
                text=True,
                capture_output=True,
                timeout=120,
            )

    def _assert_gate_accepted(self, completed, expected_target):
        self.assertEqual(completed.returncode, 2, completed.stderr)
        self.assertIn("Model not found", completed.stderr)
        self.assertIn(f"download.sh {expected_target}", completed.stderr)
        self.assertNotIn(S100P_ERROR, completed.stderr)
        self.assertNotIn(UNKNOWN_ERROR, completed.stderr)

    def _assert_gate_rejected(self, completed, expected_error):
        self.assertEqual(completed.returncode, 2, completed.stderr)
        self.assertIn(expected_error, completed.stderr)
        self.assertNotIn("Model not found", completed.stderr)

    def test_s100_identity_still_executable(self):
        for board_type in ("s100\n", "\n", None):
            with self.subTest(board_type=board_type):
                completed = self._run_launcher("s100\n", board_type)
                # "Still executable" on a host means the gate passed and
                # the launcher stopped at the explicit model-preparation
                # hint for s100 — before any cmake invocation.
                self._assert_gate_accepted(completed, "s100")

    def test_s600_identity_still_executable(self):
        completed = self._run_launcher("s600\n", "s600\n")
        self._assert_gate_accepted(completed, "s600")

    def test_s600_ignores_board_type_subdivision(self):
        # board_type only subdivides soc_name=s100 (base_soc of the
        # registered s100p target); an s600 with an s100p board_type is
        # s600, mirroring match_target.
        completed = self._run_launcher("s600\n", "rdk s100p\n")
        self._assert_gate_accepted(completed, "s600")

    def test_explicit_s100p_soc_name_rejected(self):
        completed = self._run_launcher("s100p\n", None)
        self._assert_gate_rejected(completed, S100P_ERROR)

    def test_s100_soc_with_s100p_board_type_rejected(self):
        # Both registered board_type spellings, plus mixed case with the
        # "rdk " prefix as seen on real boards.
        for board_type in ("s100p\n", "rdk s100p\n", "RDK S100P\n"):
            with self.subTest(board_type=board_type):
                completed = self._run_launcher("s100\n", board_type)
                self._assert_gate_rejected(completed, S100P_ERROR)

    def test_missing_identity_files_reported_as_unknown(self):
        completed = self._run_launcher(None, None)
        self._assert_gate_rejected(completed, UNKNOWN_ERROR)
        self.assertIn("'unknown'", completed.stderr)

    def test_unknown_soc_rejected(self):
        completed = self._run_launcher("x5\n", None)
        self._assert_gate_rejected(completed, UNKNOWN_ERROR)
        self.assertIn("'x5'", completed.stderr)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
