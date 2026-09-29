"""Exercise real native CLI/application IO with an explicit model-runner double."""

from pathlib import Path
import json
import os
import shutil
import subprocess
import tempfile
import unittest
import numpy as np

SAMPLE = Path(__file__).resolve().parents[1]
ROOT = SAMPLE.parents[2]
CPP = SAMPLE / "runtime/cpp"


class NativeCliTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler = shutil.which("c++")
        includes = [
            Path(os.environ.get("NLOHMANN_JSON_INCLUDE", "/usr/include")),
            ROOT.parent / ".coordination/asr-json/include",
        ]
        include = next(
            (p for p in includes if (p / "nlohmann/json.hpp").is_file()), None
        )
        if not compiler or not include:
            raise unittest.SkipTest(
                "C++17/nlohmann headers missing: native CLI not-run"
            )
        cls.workspace = tempfile.TemporaryDirectory(prefix="himloco-cli-")
        cls.binary = Path(cls.workspace.name) / "cli_fixture"
        argv = [
            compiler,
            "-std=c++17",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-I",
            str(CPP),
            "-I",
            str(ROOT / "samples/_shared/cpp"),
            "-I",
            str(include),
        ]
        argv += [
            str(CPP / f)
            for f in [
                "main.cc",
                "cli_io.cc",
                "application.cc",
                "policy.cc",
                "tests/cli_fixture_runner.cc",
            ]
        ]
        run = subprocess.run(
            argv + ["-o", str(cls.binary)], capture_output=True, text=True
        )
        if run.returncode:
            cls.workspace.cleanup()
            raise AssertionError(run.stderr)

    @classmethod
    def tearDownClass(cls):
        cls.workspace.cleanup()

    def call(self, *args, fail=False):
        env = os.environ.copy()
        if fail:
            env["HIMLOCO_FIXTURE_FAIL_AFTER"] = "3"
        return subprocess.run(
            [str(self.binary), *map(str, args)],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
        )

    def test_complete_outputs_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "result"
            run = self.call("--output-dir", out, "--warmup", "2")
            self.assertEqual(run.returncode, 0, run.stderr)
            report = json.loads((out / "report.json").read_text())
            self.assertEqual(
                (report["status"], report["sample_count"], report["warmup_completed"]),
                ("completed", 21, 2),
            )
            self.assertEqual(len(report["records"]), 21)
            for record in report["records"]:
                values = np.fromfile(record["output_file"], dtype="<f4")
                self.assertEqual(values.tolist(), list(range(12)))
                self.assertEqual(len(record["input_sha256"]), 64)
            before = (out / "report.json").read_bytes()
            self.assertEqual(self.call("--output-dir", out).returncode, 2)
            self.assertEqual((out / "report.json").read_bytes(), before)

    def test_partial_failure_report(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "partial"
            run = self.call("--output-dir", out, "--warmup", "2", fail=True)
            self.assertEqual(run.returncode, 2)
            report = json.loads((out / "report.json").read_text())
            self.assertEqual(
                (
                    report["status"],
                    report["sample_count"],
                    report["current_source_index"],
                ),
                ("failed", 1, 1),
            )
            self.assertNotIn("latency_ms", report)
            self.assertEqual(len(list(out.glob("*.bin"))), 1)

    def test_manifest_digest_and_cli_rejections(self):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory) / "data"
            shutil.copytree(SAMPLE / "test_data", data)
            (data / "obs_history/000000.bin").write_bytes(bytes(1080))
            out = Path(directory) / "result"
            run = self.call("--input-path", data / "obs_history", "--output-dir", out)
            self.assertEqual(run.returncode, 2)
            self.assertIn("digest", run.stderr.lower())
            # Discovery binds the manifest; bytes are checked at use and retained in failed report.
            if out.exists():
                self.assertEqual(
                    json.loads((out / "report.json").read_text())["status"], "failed"
                )
            for args in [
                ("--warmup", "-1"),
                ("--priority", "256"),
                ("--target", "s100"),
                ("--warmup", "1", "--warmup", "2"),
                ("--unknown", "1"),
            ]:
                self.assertEqual(self.call(*args).returncode, 2)
            self.assertEqual(self.call("--help").returncode, 0)
