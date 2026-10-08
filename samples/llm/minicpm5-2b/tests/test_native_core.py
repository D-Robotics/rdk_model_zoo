"""Host checks for the MiniCPM native cores.

Production sources are compiled against the SDK test doubles in
``tests/native/fixtures`` and executed. The doubles are clearly marked and are
not the vendor SDK: no model weights, no BPU device and no board is involved.
"""

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

TESTS = Path(__file__).resolve().parent
SAMPLE = TESTS.parent
REPO = SAMPLE.parents[2]
NATIVE = TESTS / "native"
CXX = os.environ.get("MINICPM_CXX", "c++")
HOST_VALIDATION = REPO / "scripts" / "tools" / "host_validation"
if str(HOST_VALIDATION) not in sys.path:
    sys.path.insert(0, str(HOST_VALIDATION))

from native_dependencies import (  # noqa: E402
    NativeDependencyMissing,
    json_include_dir,
)

LEGACY_INC = SAMPLE / "runtime/legacy/inc"
LEGACY_SRC = [
    SAMPLE / "runtime/legacy/src/minicpm5.cc",
    SAMPLE / "runtime/legacy/src/chat_template.cc",
]
CPP_INC = SAMPLE / "runtime/cpp/inc"
CPP_SRC = [
    SAMPLE / "runtime/cpp/src/minicpm5.cc",
    SAMPLE / "runtime/cpp/src/runtime_config.cc",
]

FIXTURES = NATIVE / "fixtures"
# -fno-elide-constructors keeps the PreparedRequest ownership regression
# honest: no return-value optimization can mask a rebinding bug.
EXTRA_FLAGS = ["-fno-elide-constructors"]

# S600 drivers compile runtime_config.cc, which includes nlohmann/json.hpp
# (MINICPM_JSON_INCLUDE override, pkg-config or standard system discovery).
JSON_DRIVERS = frozenset({"s600_config_cleanup", "s600_metrics", "s600_stages"})

DRIVERS = {
    "legacy_lifecycle": (
        [NATIVE / "legacy_lifecycle.cpp"],
        LEGACY_SRC,
        [FIXTURES, LEGACY_INC],
    ),
    "legacy_request": (
        [NATIVE / "legacy_request.cpp"],
        LEGACY_SRC,
        [FIXTURES, LEGACY_INC],
    ),
    "legacy_prepared_ownership": (
        [NATIVE / "legacy_prepared_ownership.cpp"],
        LEGACY_SRC,
        [FIXTURES, LEGACY_INC],
    ),
    "legacy_stream_sink": (
        [NATIVE / "legacy_stream_sink.cpp"],
        LEGACY_SRC,
        [FIXTURES, LEGACY_INC],
    ),
    "s600_config_cleanup": (
        [NATIVE / "s600_config_cleanup.cpp"],
        CPP_SRC,
        [FIXTURES, CPP_INC],
    ),
    "s600_metrics": (
        [NATIVE / "s600_metrics.cpp"],
        CPP_SRC,
        [FIXTURES, CPP_INC],
    ),
    "s600_stages": (
        [NATIVE / "s600_stages.cpp"],
        CPP_SRC,
        [FIXTURES, CPP_INC],
    ),
}


def resolve_json_include():
    """nlohmann include dir: ``MINICPM_JSON_INCLUDE`` override or discovery.

    Returns ``(include, None)`` on success or ``(None, reason)`` when no
    headers are installed, so only the JSON-dependent S600 drivers skip; an
    invalid override raises and fails the run.
    """
    try:
        return json_include_dir(os.environ.get("MINICPM_JSON_INCLUDE")), None
    except NativeDependencyMissing as error:
        return None, f"{error}; S600 drivers not-run"


def compile_driver(name, build_dir, json_include=None):
    driver, sources, includes = DRIVERS[name]
    if name in JSON_DRIVERS:
        includes = [*includes, json_include]
    binary = build_dir / name
    command = [
        CXX,
        "-std=c++17",
        "-Wall",
        "-Wextra",
        "-Werror",
        *EXTRA_FLAGS,
        *sum((["-I", str(include)] for include in includes), []),
        *[str(path) for path in driver],
        *[str(source) for source in sources],
        "-o",
        str(binary),
    ]
    done = subprocess.run(command, capture_output=True, text=True)
    if done.returncode:
        raise RuntimeError(f"compile {name} failed:\n{done.stderr}")
    return binary


class NativeCoreTests(unittest.TestCase):
    def setUp(self):
        self.build = tempfile.TemporaryDirectory()
        self.addCleanup(self.build.cleanup)
        self.json_include, self.json_skip = resolve_json_include()
        self.binaries = {}
        for name in DRIVERS:
            if name in JSON_DRIVERS and self.json_include is None:
                continue
            self.binaries[name] = compile_driver(
                name, Path(self.build.name), self.json_include
            )

    def run_driver(self, name):
        done = subprocess.run(
            [str(self.binaries[name])], capture_output=True, text=True, timeout=120
        )
        self.assertEqual(done.returncode, 0, f"{name} stderr:\n{done.stderr}")
        self.assertIn("OK", done.stdout)

    def require_json_driver(self, name):
        if name not in self.binaries:
            self.skipTest(self.json_skip)

    def test_legacy_lifecycle_single_use(self):
        self.run_driver("legacy_lifecycle")

    def test_legacy_request_stages(self):
        self.run_driver("legacy_request")

    def test_legacy_prepared_ownership(self):
        self.run_driver("legacy_prepared_ownership")

    def test_legacy_stream_sink(self):
        self.run_driver("legacy_stream_sink")

    def test_s600_config_cleanup(self):
        self.require_json_driver("s600_config_cleanup")
        self.run_driver("s600_config_cleanup")

    def test_s600_metric_validation(self):
        self.require_json_driver("s600_metrics")
        self.run_driver("s600_metrics")

    def test_s600_stage_boundaries(self):
        self.require_json_driver("s600_stages")
        self.run_driver("s600_stages")

    def test_drivers_preserve_parent_temp_sentinel(self):
        """CORE-R4: a hostile TMPDIR with a pre-existing model/ sentinel must
        survive every driver; the binaries are invoked directly, without the
        Python wrapper, and all RED demonstrations use this fresh directory."""
        if set(self.binaries) != set(DRIVERS):
            # Reduced driver coverage (missing nlohmann headers) must be
            # visible, not a partial pass.
            self.skipTest(self.json_skip)
        for name in self.binaries:
            with self.subTest(driver=name), tempfile.TemporaryDirectory() as parent:
                model_dir = Path(parent) / "model"
                model_dir.mkdir()
                sentinel = model_dir / "sentinel.txt"
                sentinel.write_text("do-not-delete")
                done = subprocess.run(
                    [str(self.binaries[name])],
                    env=dict(os.environ, TMPDIR=parent),
                    capture_output=True,
                    text=True,
                    timeout=180,
                )
                self.assertEqual(done.returncode, 0, f"{name} stderr:\n{done.stderr}")
                self.assertTrue(sentinel.exists(), f"{name} deleted the sentinel")
                self.assertEqual(
                    sentinel.read_text(), "do-not-delete", f"{name} overwrote it"
                )

    def test_s600_drivers_isolate_concurrent_runs(self):
        """CORE-R4: simultaneous driver instances sharing one TMPDIR must not
        collide on fixed fixture names."""
        names = (
            "s600_config_cleanup",
            "s600_metrics",
            "s600_stages",
            "s600_metrics",
            "s600_stages",
        )
        missing = [name for name in names if name not in self.binaries]
        if missing:
            self.skipTest(self.json_skip)
        procs = [
            (
                name,
                subprocess.Popen(
                    [str(self.binaries[name])],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                ),
            )
            for name in names
        ]
        for name, proc in procs:
            out, err = proc.communicate(timeout=300)
            self.assertEqual(proc.returncode, 0, f"{name} stderr:\n{err}")
            self.assertIn("OK", out)


class LegacyCliTests(unittest.TestCase):
    """The real main.cc against the double: RESULT line and exit statuses."""

    @classmethod
    def setUpClass(cls):
        cls.build = tempfile.TemporaryDirectory()
        driver, sources, includes = (
            [SAMPLE / "runtime/legacy/src/main.cc"],
            LEGACY_SRC,
            [FIXTURES, LEGACY_INC],
        )
        cls.binary = Path(cls.build.name) / "legacy_main"
        command = [
            CXX,
            "-std=c++17",
            "-Wall",
            "-Wextra",
            "-Werror",
            *EXTRA_FLAGS,
            *sum((["-I", str(include)] for include in includes), []),
            *[str(path) for path in driver + sources],
            "-o",
            str(cls.binary),
        ]
        done = subprocess.run(command, capture_output=True, text=True)
        if done.returncode:
            raise RuntimeError(f"compile legacy CLI failed:\n{done.stderr}")

    @classmethod
    def tearDownClass(cls):
        cls.build.cleanup()

    def run_cli(self, *arguments):
        return subprocess.run(
            [str(self.binary), *arguments], capture_output=True, text=True, timeout=120
        )

    def write_template(self, directory, contents="template"):
        path = directory / "chat.jinja"
        path.write_text(contents)
        return path

    def test_result_line_and_streamed_text(self):
        with tempfile.TemporaryDirectory() as directory:
            template = self.write_template(Path(directory))
            done = self.run_cli(
                "--model-path",
                str(Path(directory) / "model.hbm"),
                "--tokenizer-path",
                str(Path(directory) / "tokenizer"),
                "--template-path",
                str(template),
            )
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertIn("你好，很高兴认识你。", done.stdout)
        self.assertTrue(
            done.stdout.endswith("RESULT status=0 ended=1 failed=0 destroy=0\n"),
            done.stdout,
        )

    def test_missing_template_fails_with_source_message(self):
        with tempfile.TemporaryDirectory() as directory:
            done = self.run_cli(
                "--model-path",
                "m",
                "--tokenizer-path",
                "t",
                "--template-path",
                str(Path(directory) / "absent.jinja"),
            )
        self.assertEqual(done.returncode, 1)
        self.assertIn("Cannot open chat template", done.stderr)

    def test_help_and_argument_errors(self):
        self.assertEqual(self.run_cli("--help").returncode, 0)
        self.assertEqual(self.run_cli("--bogus", "x").returncode, 1)
        self.assertEqual(self.run_cli("--prompt").returncode, 1)


if __name__ == "__main__":
    unittest.main()
