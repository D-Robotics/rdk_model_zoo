"""SDK-free command checks for the MobileNetV3 entrypoint."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parents[4]
MAIN = ROOT / "samples" / "vision" / "mobilenetv3" / "runtime" / "python" / "main.py"


class EntrypointTests(unittest.TestCase):
    def _run(self, *args):
        env = dict(os.environ)
        env.pop("PYTHONPATH", None)
        return subprocess.run(
            [sys.executable, str(MAIN), *args],
            cwd=str(ROOT),
            env=env,
            text=True,
            capture_output=True,
        )

    def test_help_does_not_need_board_sdk(self):
        completed = self._run("--help")
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--target", completed.stdout)

    def test_list_models_publishes_every_manifest_filename(self):
        completed = self._run("--list-models", "--target", "auto")
        self.assertEqual(completed.returncode, 0, completed.stderr)
        for filename in ('MobileNetV3_224x224_nv12.bin', 's100/mobilenetv3_224x224_nv12.hbm', 's600/mobilenetv3_224x224_nv12.hbm'):
            self.assertIn(filename, completed.stdout)

    def test_dry_run_per_target_reports_source_contract(self):
        for target, variant, expected in (('x5', 'mobilenetv3', {'protocol': 'packed_nv12', 'geometry': '224x224', 'policy': 'softmax'}), ('s100', 'mobilenetv3', {'protocol': 'split_nv12', 'geometry': '224x224', 'policy': 'softmax'}), ('s600', 'mobilenetv3', {'protocol': 'split_nv12', 'geometry': '224x224', 'policy': 'softmax'})):
            completed = self._run(
                "--dry-run", "--target", target
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn(f"input_protocol: {expected['protocol']}", completed.stdout)
            self.assertIn(f"input_geometry: {expected['geometry']}", completed.stdout)
            self.assertIn(
                f"output_score_policy: {expected['policy']}", completed.stdout
            )
            self.assertIn("No model is downloaded", completed.stdout)
            self.assertNotIn("hbm_runtime", completed.stdout)

    def test_missing_model_is_a_visible_error_without_implicit_download(self):
        completed = self._run(
            "--target", "x5",
            "--asset-id", "x5:mobilenetv3:MobileNetV3_224x224_nv12.bin",
            "--model-path", str(ROOT / "missing-mobilenetv3-model.bin"),
        )
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("model file not found", completed.stderr.lower())


if __name__ == "__main__":
    unittest.main()
