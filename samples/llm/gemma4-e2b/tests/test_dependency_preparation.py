"""Dependency installer tests use local Git repositories and fake Rust only."""

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "third_party/install_tokenizers_cpp.sh"
PIN = "c586c52f93f7b060753bd2388eb96a105cb7374d"
URL = "https://github.com/mlc-ai/tokenizers-cpp.git"


class DependencyPreparationTests(unittest.TestCase):
    def setUp(self):
        if not shutil.which("git"):
            self.skipTest("Git unavailable")
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root / "upstream"
        self.source.mkdir()
        self.env = dict(
            os.environ,
            HOME=str(self.root),
            GIT_CONFIG_NOSYSTEM="1",
            GIT_CONFIG_GLOBAL="/dev/null",
        )
        for key in tuple(self.env):
            if key.startswith("GIT_") and key not in (
                "GIT_CONFIG_NOSYSTEM",
                "GIT_CONFIG_GLOBAL",
            ):
                self.env.pop(key)

        def git(*args):
            return subprocess.check_output(
                ["git", "-C", str(self.source), *args],
                env=self.env,
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()

        self.git = git
        git("init")
        for name in [
            "CMakeLists.txt",
            "msgpack/CMakeLists.txt",
            "sentencepiece/CMakeLists.txt",
            "rust/Cargo.lock",
        ]:
            path = self.source / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("version = 4\n" if "Cargo.lock" in name else "# fixture\n")
        git("add", ".")
        git(
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "-m",
            "fixture",
        )
        self.pin = git("rev-parse", "HEAD")
        self.script = self.root / "install.sh"
        self.script.write_text(
            SCRIPT.read_text().replace(PIN, self.pin).replace(URL, str(self.source))
        )
        self.bin = self.root / "bin"
        self.bin.mkdir()
        for name, body in [
            ("rustc", "echo rustc 1.80.0"),
            ("cargo", "echo cargo 1.80.0"),
            ("curl", "echo unexpected-network >&2; exit 97"),
        ]:
            p = self.bin / name
            p.write_text("#!/bin/bash\n" + body + "\n")
            p.chmod(0o755)
        self.env["PATH"] = str(self.bin) + ":" + os.environ["PATH"]
        self.dest = self.root / "tokenizers-cpp"

    def run_script(self, *args):
        return subprocess.run(
            ["bash", str(self.script), *args],
            env=self.env,
            capture_output=True,
            text=True,
        )

    def test_incomplete_existing_directory_is_preserved(self):
        self.dest.mkdir()
        (self.dest / "my-work").write_text("keep")
        result = self.run_script()
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual((self.dest / "my-work").read_text(), "keep")

    def test_local_pin_install_and_reuse(self):
        result = self.run_script()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(
            subprocess.check_output(
                ["git", "-C", str(self.dest), "rev-parse", "HEAD"], text=True
            ).strip(),
            self.pin,
        )
        self.assertEqual((self.dest / "rust/Cargo.lock").read_text(), "version = 3\n")
        result = self.run_script()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_modified_existing_checkout_is_preserved_and_rejected(self):
        self.assertEqual(self.run_script().returncode, 0)
        p = self.dest / "CMakeLists.txt"
        p.write_text("my edits")
        result = self.run_script()
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(p.read_text(), "my edits")

    def test_old_rust_is_rejected_without_installation(self):
        (self.bin / "rustc").write_text("#!/bin/bash\necho rustc 1.79.0\n")
        result = self.run_script()
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn("unexpected-network", result.stderr)
        self.assertFalse(self.dest.exists())

    def test_wrong_commit_is_preserved(self):
        self.assertEqual(self.run_script().returncode, 0)
        subprocess.check_call(
            [
                "git",
                "-C",
                str(self.dest),
                "-c",
                "user.name=Fixture",
                "-c",
                "user.email=fixture@example.invalid",
                "commit",
                "--allow-empty",
                "-m",
                "local commit",
            ],
            env=self.env,
            stdout=subprocess.DEVNULL,
        )
        result = self.run_script()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("commit mismatch", result.stderr)

    def test_failed_clone_leaves_no_destination_or_stage(self):
        shutil.rmtree(self.source)
        result = self.run_script()
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(self.dest.exists())
        self.assertEqual(list(self.root.glob(".tokenizers-stage.*")), [])

    def test_dirty_pinned_submodule_is_rejected(self):
        # Point a real gitlink at the earlier local fixture commit, then pin it.
        self.env["GIT_ALLOW_PROTOCOL"] = "file"
        self.git("rm", "-r", "msgpack")
        self.git("submodule", "add", str(self.source), "msgpack")
        self.git(
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "-am",
            "add local submodule",
        )
        new_pin = self.git("rev-parse", "HEAD")
        self.script.write_text(self.script.read_text().replace(self.pin, new_pin))
        result = self.run_script()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        file = self.dest / "msgpack/CMakeLists.txt"
        file.write_text("my submodule edit")
        result = self.run_script()
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(file.read_text(), "my submodule edit")

    def test_preview_has_no_writes(self):
        result = self.run_script("--dry-run")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(self.dest.exists())
        self.assertIn(self.pin, result.stdout)


if __name__ == "__main__":
    unittest.main()
