"""Exercise real native CLI/application IO with an explicit model-runner double."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
import numpy as np

SAMPLE = Path(__file__).resolve().parents[1]
ROOT = SAMPLE.parents[2]
CPP = SAMPLE / "runtime/cpp"
HOST_VALIDATION = ROOT / "tools" / "host_validation"
if str(HOST_VALIDATION) not in sys.path:
    sys.path.insert(0, str(HOST_VALIDATION))

from native_dependencies import (  # noqa: E402
    NativeDependencyMissing,
    NativeDependencyOverrideError,
    json_include_dir,
)

NO_PKG_CONFIG = lambda package, mode, environ: None  # noqa: E731


def resolve_json_include(environ=None, **discovery):
    """nlohmann include dir: ``NLOHMANN_JSON_INCLUDE`` override or discovery.

    Standard discovery (pkg-config / system include roots) lives in
    ``tools/host_validation/native_dependencies.py``. An invalid override
    fails the run; a host without any nlohmann headers skips these checks
    explicitly. There is no fallback to a personal directory.
    """
    environ = os.environ if environ is None else environ
    override = str(environ.get("NLOHMANN_JSON_INCLUDE", "")).strip() or None
    try:
        return json_include_dir(override, environ=environ, **discovery)
    except NativeDependencyMissing as error:
        raise unittest.SkipTest(
            f"nlohmann-json headers missing ({error}); native CLI not-run"
        ) from error


class NativeCliTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler = shutil.which("c++")
        if not compiler:
            raise unittest.SkipTest("C++17 compiler unavailable; native CLI not-run")
        include = resolve_json_include()
        cls.workspace = tempfile.TemporaryDirectory(prefix="himloco-cli-")
        try:
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
                raise AssertionError(run.stderr)
        except BaseException:
            # Never leak the build workspace when class construction fails.
            cls.workspace.cleanup()
            raise

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


class JsonIncludeResolutionTests(unittest.TestCase):
    """Missing vs invalid nlohmann dependency is a skip vs a failure.

    Regression for the former personal-directory fallback: an explicit
    ``NLOHMANN_JSON_INCLUDE`` that does not carry ``nlohmann/json.hpp`` must
    fail the run, not silently fall back to another machine-specific
    directory; only a genuinely header-less host skips.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="himloco-json-")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_valid_override_is_respected(self):
        include = self.root / "json-include"
        (include / "nlohmann").mkdir(parents=True)
        (include / "nlohmann" / "json.hpp").write_text("// fixture header\n")
        self.assertEqual(
            resolve_json_include(environ={"NLOHMANN_JSON_INCLUDE": str(include)}),
            include,
        )

    def test_override_without_header_fails_instead_of_falling_back(self):
        empty = self.root / "empty-include"
        empty.mkdir()
        with self.assertRaises(NativeDependencyOverrideError):
            resolve_json_include(environ={"NLOHMANN_JSON_INCLUDE": str(empty)})

    def test_override_pointing_at_missing_directory_fails(self):
        with self.assertRaises(NativeDependencyOverrideError):
            resolve_json_include(
                environ={"NLOHMANN_JSON_INCLUDE": str(self.root / "absent")}
            )

    def test_missing_headers_skip_explicitly(self):
        with self.assertRaises(unittest.SkipTest) as raised:
            resolve_json_include(
                environ={},
                pkg_config=NO_PKG_CONFIG,
                include_roots=(),
            )
        self.assertIn("not-run", str(raised.exception))
