"""SDK-free command checks for the EfficientNet entrypoint."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parents[4]
MAIN = ROOT / "samples" / "vision" / "efficientnet" / "runtime" / "python" / "main.py"

PUBLISHED_FILENAMES = (
    "EfficientNet_B2_224x224_nv12.bin",
    "EfficientNet_B3_224x224_nv12.bin",
    "EfficientNet_B4_224x224_nv12.bin",
    "s100/efficientnet_lite0_224x224_nv12.hbm",
    "s100/efficientnet_lite1_240x240_nv12.hbm",
    "s100/efficientnet_lite2_260x260_nv12.hbm",
    "s100/efficientnet_lite3_300x300_nv12.hbm",
    "s100/efficientnet_lite4_380x380_nv12.hbm",
    "s600/efficientnet_lite0_224x224_nv12.hbm",
    "s600/efficientnet_lite1_240x240_nv12.hbm",
    "s600/efficientnet_lite2_260x260_nv12.hbm",
    "s600/efficientnet_lite3_300x300_nv12.hbm",
    "s600/efficientnet_lite4_380x380_nv12.hbm",
)


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
        self.assertIn("--variant", completed.stdout)

    def test_list_models_publishes_every_manifest_filename(self):
        completed = self._run("--list-models", "--target", "auto")
        self.assertEqual(completed.returncode, 0, completed.stderr)
        for filename in PUBLISHED_FILENAMES:
            self.assertIn(filename, completed.stdout, filename)

    def test_dry_run_reports_source_contract_per_target_and_variant(self):
        cases = (
            ("x5", "b2", {"protocol": "packed_nv12", "geometry": "224x224"}),
            ("x5", "b4", {"protocol": "packed_nv12", "geometry": "224x224"}),
            ("s100", "lite0", {"protocol": "split_nv12", "geometry": "224x224"}),
            ("s100", "lite1", {"protocol": "split_nv12", "geometry": "240x240"}),
            ("s100", "lite4", {"protocol": "split_nv12", "geometry": "380x380"}),
            ("s600", "lite3", {"protocol": "split_nv12", "geometry": "300x300"}),
        )
        for target, variant, expected in cases:
            with self.subTest(target=target, variant=variant):
                completed = self._run(
                    "--dry-run", "--target", target, "--variant", variant
                )
                self.assertEqual(completed.returncode, 0, completed.stderr)
                self.assertIn(f"input_protocol: {expected['protocol']}", completed.stdout)
                self.assertIn(f"input_geometry: {expected['geometry']}", completed.stdout)
                self.assertIn("output_score_policy: softmax", completed.stdout)
                self.assertIn("No model is downloaded", completed.stdout)
                self.assertNotIn("hbm_runtime", completed.stdout)

    def test_missing_model_is_a_visible_error_without_implicit_download(self):
        completed = self._run(
            "--target", "x5",
            "--asset-id", "x5:efficientnet:EfficientNet_B2_224x224_nv12.bin",
            "--model-path", str(ROOT / "missing-efficientnet-model.bin"),
        )
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("model file not found", completed.stderr.lower())


if __name__ == "__main__":
    unittest.main()
