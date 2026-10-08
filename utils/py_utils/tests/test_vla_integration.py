"""Pinned third-party VLA integration is checked without executing its code."""

import configparser
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[3]
INTEGRATIONS = ROOT / "samples/vla/integrations.json"


def declared_gitlink_paths(registry=None):
    """The pinned integration gitlink paths declared by ``integrations.json``."""
    if registry is None:
        registry = json.loads(INTEGRATIONS.read_text())
    return {"samples/vla/" + name for name in registry}


def shadow_gitlinks(listing, declared):
    """Gitlink paths in ``git ls-files --stage`` output outside ``declared``.

    Gitlink rows carry mode ``160000``. The two pinned VLA integrations are
    legitimate, deliberately-present gitlinks (see the sibling pin test), so
    only paths outside ``declared`` are shadow entries.
    """
    shadows = []
    for line in listing.splitlines():
        metadata, separator, path = line.partition("\t")
        if not separator:
            continue
        if metadata.split(" ", 1)[0] == "160000" and path not in declared:
            shadows.append(path)
    return sorted(shadows)


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
        # Gitlinks are legal exactly for the pinned integrations declared in
        # integrations.json (whose pins the sibling test verifies row-exact);
        # the historical platforms/ tree whose stray gitlinks this check
        # originally guarded against was removed with the migration closeout,
        # so any other gitlink under the active samples/ tree is an
        # unexpected shadow.
        listing = subprocess.check_output(
            ["git", "ls-files", "--stage", "--", "samples/"],
            cwd=ROOT,
            text=True,
        )
        shadows = shadow_gitlinks(listing, declared_gitlink_paths(registry))
        self.assertFalse(shadows, f"unexpected gitlink(s) under samples/: {shadows}")


class ShadowGitlinkGuardTests(unittest.TestCase):
    """The parent-tree guard: declared pins pass, any other gitlink fails."""

    DECLARED = {"samples/vla/act", "samples/vla/pi0"}

    def test_legal_listing_with_only_declared_pins_passes(self):
        listing = (
            "100644 5c0ae19 0\tsamples/README.md\n"
            "160000 326ea043be204de25223d95c7d918efe8672dc66 0\tsamples/vla/act\n"
            "160000 a32de276bc1681a2b1531012de111eaa1c16acb6 0\tsamples/vla/pi0\n"
        )
        self.assertEqual(shadow_gitlinks(listing, self.DECLARED), [])

    def test_unexpected_gitlink_elsewhere_under_samples_is_a_shadow(self):
        listing = (
            "160000 326ea043be204de25223d95c7d918efe8672dc66 0\tsamples/vla/act\n"
            "160000 0000000000000000000000000000000000000001 0\tsamples/robotics/rogue\n"
        )
        self.assertEqual(
            shadow_gitlinks(listing, self.DECLARED), ["samples/robotics/rogue"]
        )

    def test_unexpected_gitlink_nested_under_vla_is_a_shadow(self):
        listing = (
            "160000 326ea043be204de25223d95c7d918efe8672dc66 0\tsamples/vla/act\n"
            "160000 0000000000000000000000000000000000000002 0\tsamples/vla/extra\n"
        )
        self.assertEqual(shadow_gitlinks(listing, self.DECLARED), ["samples/vla/extra"])

    def test_ordinary_files_and_directories_are_never_shadows(self):
        listing = (
            "100644 5c0ae19 0\tsamples/vla/README.md\n"
            "100755 5c0ae19 0\tsamples/vla/guides/act.md\n"
            "040000 5c0ae19 0\tsamples/vla/other\n"
        )
        self.assertEqual(shadow_gitlinks(listing, self.DECLARED), [])

    def test_empty_listing_passes(self):
        self.assertEqual(shadow_gitlinks("", self.DECLARED), [])

    def test_real_git_listing_from_isolated_index_fixture(self):
        # End-to-end parsing against real `git ls-files --stage` output, built
        # with git init/update-index inside a temporary fixture checkout only;
        # the parent repository's index is never touched.
        with tempfile.TemporaryDirectory(prefix="vla-gitlink-fixture-") as tmp:
            fixture = Path(tmp) / "fixture-repo"
            fixture.mkdir()

            def git(*args):
                return subprocess.check_output(
                    ["git", *args],
                    cwd=fixture,
                    text=True,
                    stderr=subprocess.DEVNULL,
                )

            git("init")
            (fixture / "samples").mkdir()
            (fixture / "samples" / "README.md").write_text("fixture\n")
            git("add", "samples/README.md")
            git(
                "update-index",
                "--add",
                "--cacheinfo",
                "160000,326ea043be204de25223d95c7d918efe8672dc66,samples/vla/act",
            )
            self.assertEqual(
                shadow_gitlinks(git("ls-files", "--stage", "--", "samples/"), self.DECLARED),
                [],
            )
            git(
                "update-index",
                "--add",
                "--cacheinfo",
                "160000,0000000000000000000000000000000000000003,samples/robotics/shadow",
            )
            self.assertEqual(
                shadow_gitlinks(git("ls-files", "--stage", "--", "samples/"), self.DECLARED),
                ["samples/robotics/shadow"],
            )
