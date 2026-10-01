"""Pinned third-party VLA integration is checked without executing its code."""

import configparser
import json
from pathlib import Path
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[3]


class VlaIntegrationTests(unittest.TestCase):
    def test_gitlinks_and_module_urls_match_pins(self):
        registry = json.loads((ROOT / "samples/vla/integrations.json").read_text())
        modules = configparser.ConfigParser()
        modules.read(ROOT / ".gitmodules")
        self.assertEqual(set(registry), {"act", "pi0"})
        self.assertNotEqual(registry["act"]["commit"], registry["pi0"]["commit"])
        for name, item in registry.items():
            with self.subTest(name=name):
                path = "samples/vla/" + name
                stage = subprocess.check_output(
                    ["git", "ls-files", "--stage", "--", path], cwd=ROOT, text=True
                )
                self.assertEqual(stage.strip(), f"160000 {item['commit']} 0\t{path}")
                section = f'submodule "{path}"'
                self.assertEqual(modules[section]["path"], path)
                self.assertEqual(modules[section]["url"], item["url"])
                for guide in item["guides"]:
                    text = (ROOT / guide).read_text()
                    self.assertIn(item["commit"], text)
                    self.assertIn("git submodule update --init --checkout", text)
                self.assertFalse(item["board_tested_this_migration"])

    def test_manual_assets_and_no_shadow_gitlinks(self):
        import yaml

        registry = json.loads((ROOT / "samples/vla/integrations.json").read_text())
        models = yaml.safe_load((ROOT / "docs/release/s/models.yaml").read_text())[
            "models"
        ]
        for name in registry:
            row = next(row for row in models if row["id"] == name)
            self.assertEqual(row["sample_path"], "samples/vla/" + name)
            self.assertEqual(row["availability"], "manual")
            self.assertEqual(row["assets"], [])
        # The historical platforms/s tree (whose stray gitlinks this check
        # originally guarded against) was removed with the migration
        # closeout; verify no gitlink shadows exist anywhere under the
        # active tree instead.
        entries = subprocess.check_output(
            ["git", "ls-files", "--stage", "--", "samples/"],
            cwd=ROOT,
            text=True,
        )
        self.assertFalse(
            any(line.startswith("160000 ") for line in entries.splitlines())
        )
