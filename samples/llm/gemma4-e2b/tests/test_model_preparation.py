"""Downloader orchestration only; fake wget writes bytes, no network/models."""

import os
from pathlib import Path
import subprocess
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "model/download_model.sh"


class ModelPreparationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.home = self.root / "artifacts"
        self.bin = self.root / "bin"
        self.bin.mkdir()
        wget = self.bin / "wget"
        wget.write_text(
            '#!/bin/bash\nprintf "%s\\n" "$*" >> "$CALLS"\nif [[ "${EMPTY:-}" != 1 ]]; then printf fixture > "$3"; else : > "$3"; fi\n'
        )
        wget.chmod(0o755)
        self.env = {k: v for k, v in os.environ.items() if not k.startswith("GEMMA4_")}
        self.env.update(
            GEMMA4_HOME=str(self.home),
            PATH=str(self.bin) + ":" + os.environ["PATH"],
            CALLS=str(self.root / "calls"),
        )

    def run_script(self, *args, **env):
        return subprocess.run(
            ["bash", str(SCRIPT), *args],
            env={**self.env, **env},
            capture_output=True,
            text=True,
        )

    def test_requires_explicit_target_without_side_effects(self):
        result = self.run_script()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("GEMMA4_SOC", result.stdout + result.stderr)
        self.assertFalse(self.home.exists())

    def test_preview_does_not_create_or_download(self):
        for target, archive in (("s100p", "rdk_s100"), ("s600", "rdk_s600")):
            result = self.run_script("--dry-run", GEMMA4_SOC=target)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn(archive, result.stdout)
            self.assertFalse(self.home.exists())
            self.assertFalse((self.root / "calls").exists())

    def test_s100_requires_explicit_hbm_source(self):
        result = self.run_script(GEMMA4_SOC="s100")
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(self.home.exists())
        result = self.run_script(
            "--dry-run",
            GEMMA4_SOC="s100",
            GEMMA4_MODEL_BASE_URL="https://example.invalid/model",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("example.invalid/model", result.stdout)
        self.assertFalse(self.home.exists())

    def test_download_and_reuse_with_fake_transport(self):
        result = self.run_script(GEMMA4_SOC="s600")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len((self.root / "calls").read_text().splitlines()), 5)
        self.assertEqual(len(list(self.home.rglob("*.part"))), 0)
        result = self.run_script(GEMMA4_SOC="s600")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len((self.root / "calls").read_text().splitlines()), 5)

    def test_empty_transfer_is_not_published(self):
        result = self.run_script(GEMMA4_SOC="s600", EMPTY="1")
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse((self.home / "model/gemma4-e2b_vit_ptq.hbm").exists())

    def test_unknown_argument_has_no_effect(self):
        result = self.run_script("--unknown", GEMMA4_SOC="s600")
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(self.home.exists())


if __name__ == "__main__":
    unittest.main()
