# SPDX-License-Identifier: Apache-2.0
"""Fixture runner tests for ``scripts/tools/host_validation/run.py``.

These tests drove the runner implementation test-first: every scenario below
reproduced a failure against the pre-implementation state before the
implementation landed.  They build small synthetic repositories under a
temporary directory and execute the real runner entry point as a subprocess,
so discovery, isolation, skip classification, timeouts and report structure
are checked end to end without touching the repository worktree.

The synthetic repositories deliberately contain:

* two sample suites whose test modules share the same file name (isolated
  subprocesses must keep their results separate);
* a nested ``conversion/tests`` module that imports ``torch`` at import time
  (the declared optional export scope, recorded instead of erroring) and a
  Paraformer-style export-stage suite whose conditional skip is the only
  runtime framework skip allowed;
* a parent-repo VLA gitlink guard in the shared suite (executed — it never
  touches upstream ACT/Pi0 code);
* a native-only ``runtime/cpp/tests`` directory without Python tests;
* suites for failures, errors, native-prerequisite skips, zero-test
  directories, worker crashes, loader crashes and per-suite timeouts;
* source-drift scenarios: tracked/untracked content edited during the run
  with an unchanged dirty-file list, ignored artifacts that must NOT count
  as drift, gitlinks that must not be hashed, non-Git checkouts and report
  paths inside the source tree;
* CTest fixtures for default-on project flags, stage timeouts, missing
  executables, configure failures, discovery-JSON corruption, declared/run
  count mismatch and zero declared cases — including the exact safe-default
  guard fixtures for the Paraformer and HIMLoco runtime/cpp projects, the
  rejection of ``--cmake-define`` overrides that would enable their vendor
  SDK or production CLI builds, the rejection of typed CMake cache keys
  (``VAR:BOOL``/``:STRING``/``:PATH``/...) and of whitespace-padded false
  tokens (``" OFF "``, ``" false "``, ``" 0 "`` — CMake preserves the
  leading whitespace of a ``-D`` value, so a padded token cannot be
  trusted as OFF) — before any configure
  runs, with fake cmake/ctest sentinels proving no tool is invoked; scope
  accounting classifies with real CMake's own false-constant semantics
  (``""``, ``N``, ``IGNORE``, the case-insensitive named constants, the
  case-sensitive ``NOTFOUND``/``*-NOTFOUND`` spellings, and the trailing
  whitespace CMake strips itself — verified against real CMake 4.4.4
  option()/if() behaviour) any real OFF of a default-ON
  test/IO/sanitizer flag: recorded as a reduction through both the
  classifier and the CLI/report path with the raw value, so no false
  spelling can disable required flags while the run stays CI-equivalent;
* catalog fixtures for Node ``engines`` enforcement (satisfied, mismatched,
  missing, unparseable);
* a catalog-before-Python ordering fixture (the 2026-10-06 clean-clone
  defect): a fake node/npm pair whose ``npm run check`` writes the ignored
  generated catalog (``scripts/tools/catalog-publisher/dist/catalog.json``) that a
  dependent sample suite reads — no initial ``dist/`` — with the maintainer
  launched from outside the repository, ``PYTHONPATH``/``PYTHONHOME``
  cleared, the catalog section enabled and only CTest skipped; the check
  must run exactly once, before that suite, and a failing check (exit 7)
  still fails the whole run;
* sample-inventory coverage fixtures (happy path with real-shaped rows,
  deleted tests directory, extra sample, broken JSON, absent inventory,
  empty rows array and malformed/duplicate rows) — the accepted inventory
  is required proof: a checkout without it fails the gate;
* a cwd-probe repository launched from a distinct temporary directory
  outside it with ``PYTHONPATH``/``PYTHONHOME`` cleared (the independent
  clean-clone gate's shape): its suite asserts the repo-root working
  directory, reads a tracked repo-relative file and imports a repo-root
  module from a nested child interpreter — every path that broke when
  suite subprocesses inherited the caller's directory.

Running these tests requires no board SDK, no model files and no network.
Tests that build CMake fixtures need ``cmake`` (or ``ctest``) on PATH or
next to the interpreter and skip with a ``native prerequisite (cmake)
unavailable`` reason when absent — the same strict policy the runner itself
enforces.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
RUNNER = Path(__file__).resolve().parent / "run.py"
REQUIREMENTS = Path(__file__).resolve().parent / "requirements.txt"

# The pinned historical commit the real repository must carry.  A synthetic
# 40-hex string that no fixture repo contains is used for the missing-pin
# scenario.
REAL_PIN = "d2d2a4e0a898697bdfe5f68a9740a8c7d7cad57d"
ABSENT_PIN = "0123456789abcdef0123456789abcdef01234567"

# The accepted native sample inventory the runner validates coverage against.
SAMPLE_INVENTORY_RELPATH = (
    "docs/releases/unified-migration/2026-10-05-all-sample-coverage.json")

# The samples the green fixture repository carries.  The green layout ships
# an accepted inventory listing exactly these, because the maintained gate
# requires the inventory: a fixture without it fails rather than passing
# with a quiet ``not-present``.
GREEN_INVENTORY_SAMPLES = ("vision/alpha", "vision/beta", "vision/gamma",
                           "speech/paraformer", "robotics/nat")


def _write_inventory(repo: Path, samples) -> None:
    """Write an accepted sample inventory listing ``samples``."""
    _write(
        repo / SAMPLE_INVENTORY_RELPATH,
        json.dumps({"rows": [{"sample": sample} for sample in samples],
                    "filesystem_check": {}}),
    )


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(text).lstrip("\n"))


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-C", str(repo), "-c", "user.name=fixture", "-c",
         "user.email=fixture@example.com", *args],
        check=True, capture_output=True, text=True,
    )


def _contract_stub(repo: Path) -> None:
    _write(
        repo / "scripts/tools/sample_contract/check.py",
        """
        import argparse, json, sys
        parser = argparse.ArgumentParser()
        parser.add_argument("--scope"); parser.add_argument("--parser-mode")
        parser.add_argument("--report", required=True)
        args = parser.parse_args()
        with open(args.report, "w") as handle:
            json.dump({"summary": {"samples": 1, "violations": 0, "skips": 0,
                                   "exemptions_applied": 0}}, handle)
        sys.exit(0)
        """,
    )


def _green_layout(repo: Path) -> None:
    """Write a repository layout whose complete runner result is a pass."""
    # Mirror the real repository's ignore rules: caches never count as
    # drift and the catalog fixture's npm state stays invisible to Git.
    _write(
        repo / ".gitignore",
        "*__pycache__*\n"
        "scripts/tools/catalog-publisher/node_modules/\n",
    )
    # Same test module file name in two suites: results must stay isolated.
    _write(
        repo / "samples/vision/alpha/tests/test_model.py",
        """
        import unittest

        class AlphaModelTests(unittest.TestCase):
            def test_one(self):
                self.assertTrue(True)

            def test_two(self):
                self.assertEqual(1 + 1, 2)
        """,
    )
    # Native-only directory (no Python tests): recorded, not a Python suite.
    _write(
        repo / "samples/vision/alpha/runtime/cpp/tests/test_policy.cc",
        """
        int main() { return 0; }
        """,
    )
    _write(
        repo / "samples/vision/beta/tests/test_model.py",
        """
        import unittest

        class BetaModelTests(unittest.TestCase):
            def test_one(self):
                self.assertTrue(True)
        """,
    )
    # Nested export-scope suite: torch import failure is optional scope.
    # gamma mirrors the real yoloe-style shape: a top-level tests directory
    # (the inventory-visible invariant every sample carries) plus the
    # nested conversion suite.
    _write(
        repo / "samples/vision/gamma/tests/test_gamma.py",
        """
        import unittest

        class GammaTests(unittest.TestCase):
            def test_gamma(self):
                self.assertTrue(True)
        """,
    )
    _write(
        repo / "samples/vision/gamma/conversion/tests/test_export_heads.py",
        """
        import torch

        class ExportHeadTests(unittest.TestCase):
            def test_head(self):
                self.assertTrue(True)
        """,
    )
    _write(
        repo / "samples/vision/gamma/conversion/tests/test_export_entry.py",
        """
        import unittest

        class ExportEntryTests(unittest.TestCase):
            def test_entry(self):
                self.assertTrue(True)
        """,
    )
    # The real Paraformer export-stage identity: a conditional framework
    # skip that IS part of the declared optional export scope.
    _write(
        repo / "samples/speech/paraformer/tests/test_export_stages.py",
        """
        import importlib.util
        import unittest

        AVAILABLE = all(importlib.util.find_spec(name)
                        for name in ("torch", "funasr"))

        @unittest.skipUnless(AVAILABLE,
                             "Torch and FunASR export dependencies required")
        class ExportStages(unittest.TestCase):
            def test_stage(self):
                self.assertTrue(True)
        """,
    )
    # Shared suite now includes the VLA integration guard: it verifies the
    # parent repository's declared pins and never executes upstream code.
    _write(
        repo / "utils/py_utils/tests/test_shared.py",
        """
        import unittest

        class SharedTests(unittest.TestCase):
            def test_shared(self):
                self.assertTrue(True)
        """,
    )
    _write(
        repo / "utils/py_utils/tests/test_vla_integration.py",
        """
        import unittest

        class VlaIntegrationTests(unittest.TestCase):
            def test_parent_repo_pins_declared(self):
                # Parent-repo integrity only (pins/gitlinks); upstream
                # ACT/Pi0 code is never initialized or executed.
                self.assertTrue(True)
        """,
    )
    # robotics/nat carries the inventory-visible tests directory.
    _write(
        repo / "samples/robotics/nat/tests/test_policy.py",
        """
        import unittest

        class PolicyTests(unittest.TestCase):
            def test_policy(self):
                self.assertTrue(True)
        """,
    )
    _write(
        repo / "scripts/tools/board_validation/tests/test_tool.py",
        """
        import unittest

        class ToolTests(unittest.TestCase):
            def test_tool(self):
                self.assertTrue(True)
        """,
    )
    _write(
        repo / "skills/tests/test_pack.py",
        """
        import unittest

        class PackTests(unittest.TestCase):
            def test_pack(self):
                self.assertTrue(True)
        """,
    )
    _write(
        repo / "scripts/tools/sample_contract/tests/test_check.py",
        """
        import unittest

        class CheckerTests(unittest.TestCase):
            def test_checker(self):
                self.assertTrue(True)
        """,
    )
    _write(
        repo / "scripts/tools/host_validation/test_smoke.py",
        """
        import unittest

        class SmokeTests(unittest.TestCase):
            def test_smoke(self):
                self.assertTrue(True)
        """,
    )
    # The accepted sample inventory: the maintained gate requires it, so the
    # green fixture ships one that matches its five samples exactly.
    _write_inventory(repo, GREEN_INVENTORY_SAMPLES)
    _contract_stub(repo)


def _cwd_probe_layout(repo: Path) -> None:
    """A repository whose probe suite only passes at the repo root.

    Mirrors the 2026-10-05 independent clean-clone delivery run: the
    maintainer was launched from outside the checkout with an absolute
    ``--repo`` and a cleared ``PYTHONPATH``, and the suite subprocesses
    inherited the caller's directory, so repo-relative paths and child
    imports of the repo-root ``samples`` package failed while the same
    suites passed from the repository root.  The probe suite asserts its
    working directory, reads a tracked repo-relative file and imports a
    repo-root module from a nested child interpreter — none of which can
    pass from any other cwd.
    """
    # Cache directories must stay ignored: the child import writes
    # bytecode inside the fixture repository during the run.
    _write(repo / ".gitignore", "*__pycache__*\n")
    # Tracked repo-root file the probe reads through a relative path.
    _write(repo / "VERSION", "fixture-repo-1.0\n")
    _write(
        repo / "samples/vision/alpha/tests/test_model.py",
        """
        import unittest

        class AlphaModelTests(unittest.TestCase):
            def test_one(self):
                self.assertTrue(True)
        """,
    )
    # Import target for the child interpreter: resolvable only through the
    # repo root on sys.path — the exact "No module named 'samples'"
    # failure shape the shared import guard hit in the clean-clone run.
    _write(
        repo / "samples/vision/cwdprobe/child_helper.py",
        'MARKER = "child-import-ok"\n',
    )
    _write(
        repo / "samples/vision/cwdprobe/tests/test_repo_cwd.py",
        '''
        import subprocess
        import sys
        import unittest
        from pathlib import Path


        class RepoCwdTests(unittest.TestCase):
            def test_worker_cwd_is_requested_repo_root(self):
                repo_root = Path(__file__).resolve().parents[4]
                self.assertEqual(Path.cwd(), repo_root)

            def test_repo_relative_tracked_file_is_readable(self):
                self.assertEqual(
                    Path("VERSION").read_text().strip(), "fixture-repo-1.0")

            def test_child_python_imports_repo_root_module(self):
                proc = subprocess.run(
                    [sys.executable, "-c",
                     "import samples.vision.cwdprobe.child_helper as helper;"
                     "print(helper.MARKER)"],
                    capture_output=True, text=True)
                self.assertEqual(proc.returncode, 0, proc.stderr)
                self.assertEqual(proc.stdout.strip(), "child-import-ok")
        ''',
    )
    _write(
        repo / "scripts/tools/board_validation/tests/test_tool.py",
        """
        import unittest

        class ToolTests(unittest.TestCase):
            def test_tool(self):
                self.assertTrue(True)
        """,
    )
    _write(
        repo / "scripts/tools/sample_contract/tests/test_check.py",
        """
        import unittest

        class CheckerTests(unittest.TestCase):
            def test_checker(self):
                self.assertTrue(True)
        """,
    )
    _write(
        repo / "skills/tests/test_pack.py",
        """
        import unittest

        class PackTests(unittest.TestCase):
            def test_pack(self):
                self.assertTrue(True)
        """,
    )
    _write(
        repo / "scripts/tools/host_validation/test_smoke.py",
        """
        import unittest

        class SmokeTests(unittest.TestCase):
            def test_smoke(self):
                self.assertTrue(True)
        """,
    )
    _write(repo / "utils/py_utils/tests/test_shared.py",
           "import unittest\nclass SharedTests(unittest.TestCase):\n    def test_common(self): self.assertTrue(True)\n")
    _write_inventory(repo, ("vision/alpha", "vision/cwdprobe"))
    _contract_stub(repo)


#: The generated catalog build output the real ultralytics_yolo asset and
#: manifest snapshot suites (test_platform_assets/test_yolo26) read at the
#: repository root — produced only by ``npm run check``'s build stage.
CATALOG_MARKER_RELATIVE = "scripts/tools/catalog-publisher/dist/catalog.json"


def _catalog_dependent_layout(repo: Path) -> None:
    """Green repository plus a catalog package and a dependent suite.

    Mirrors the dependency the 2026-10-06 clean-clone gate exposed: the
    ultralytics_yolo asset/manifest snapshot suites read the generated
    ``scripts/tools/catalog-publisher/dist/catalog.json``, an ignored build output
    that only the catalog section's ``npm run check`` produces — a clean
    clone carries no ``dist/``, so those suites fail unless the catalog is
    built before the Python suites run.  The dependent suite reads the
    generated marker exactly the way those suites read the real catalog.
    """
    _green_layout(repo)
    # Mirror the real repository's ignore rule for the generated catalog:
    # the in-run build (like node_modules and caches) must never count as
    # source drift.
    _write(
        repo / ".gitignore",
        "*__pycache__*\n"
        "scripts/tools/catalog-publisher/node_modules/\n"
        "scripts/tools/catalog-publisher/dist/\n",
    )
    _write(
        repo / "scripts/tools/catalog-publisher/package.json",
        '{"name": "fixture-catalog", "private": true,'
        ' "engines": {"node": ">=22.12 <23"},'
        ' "scripts": {"check": "node -e \\"process.exit(0)\\""}}\n',
    )
    (repo / "scripts/tools/catalog-publisher/node_modules").mkdir(
        parents=True, exist_ok=True)
    _write_inventory(repo, GREEN_INVENTORY_SAMPLES + ("vision/yolo",))
    _write(
        repo / "samples/vision/yolo/tests/test_catalog_assets.py",
        """
        import json
        import unittest
        from pathlib import Path

        REPO_ROOT = Path(__file__).resolve().parents[4]

        class CatalogAssetTests(unittest.TestCase):
            def test_generated_catalog_exists_when_suites_run(self):
                catalog = json.loads(
                    (REPO_ROOT / "scripts/tools/catalog-publisher/dist/catalog.json")
                    .read_text(encoding="utf-8"))
                self.assertEqual(catalog["marker"], "fixture-catalog")
        """,
    )


def _init_repo(base: Path, layout) -> Path:
    repo = base / "repo"
    repo.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q", str(repo)], check=True,
                   capture_output=True)
    layout(repo)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "fixture")
    return repo


def run_runner(repo: Path, *args: str, timeout: float = 300) -> subprocess.CompletedProcess:
    proc = subprocess.run(
        [sys.executable, str(RUNNER), "--repo", str(repo),
         "--python", sys.executable, *args],
        capture_output=True, text=True, timeout=timeout,
    )
    return proc


def run_runner_outside(launch_dir: Path, repo: Path, *args: str,
                       timeout: float = 300) -> subprocess.CompletedProcess:
    """Launch the maintainer from ``launch_dir`` with ``PYTHONPATH`` and
    ``PYTHONHOME`` cleared — the shape of the independent clean-clone gate,
    which invokes the runner from outside the checkout with an absolute
    ``--repo`` and no ambient import environment."""
    env = {key: value for key, value in os.environ.items()
           if key not in ("PYTHONPATH", "PYTHONHOME")}
    return subprocess.run(
        [sys.executable, str(RUNNER), "--repo", str(repo),
         "--python", sys.executable, *args],
        capture_output=True, text=True, timeout=timeout,
        cwd=str(launch_dir), env=env,
    )


def _interpreter_tool(tool: str):
    candidate = Path(sys.executable).parent / tool
    if candidate.is_file() and os.access(candidate, os.X_OK):
        return str(candidate)
    return shutil.which(tool)


class RunnerFixtureTestCase(unittest.TestCase):
    """Shared helpers: green repository and report assertions."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory(prefix="host-validation-fixtures-")
        cls.tmp = Path(cls._tmp.name)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def green_repo(self) -> Path:
        cls = type(self)
        if not hasattr(cls, "_green_repo"):
            cls._green_repo = _init_repo(self.tmp / "green", _green_layout)
        return cls._green_repo

    def problem_repo(self) -> Path:
        cls = type(self)
        if not hasattr(cls, "_problem_repo"):
            cls._problem_repo = self._build_problem_repo()
        return cls._problem_repo

    def load_report(self, proc: subprocess.CompletedProcess,
                    report: Path) -> dict:
        self.assertEqual(
            proc.returncode, 0,
            f"runner failed:\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}",
        )
        self.assertTrue(report.is_file(), "report file missing")
        return json.loads(report.read_text())

    def suite_by_dir(self, report: dict, directory: str) -> dict:
        matches = [s for s in report["python_suites"]["suites"]
                   if s["dir"] == directory]
        self.assertEqual(len(matches), 1, f"suite {directory} not reported once")
        return matches[0]


class GreenRepositoryTests(RunnerFixtureTestCase):
    """A complete green repository passes with an accurate report."""

    def test_green_repository_passes_with_report_and_artifacts(self):
        repo = self.green_repo()
        out = self.tmp / "green-reports"
        report = out / "report.json"
        proc = run_runner(
            repo, "--report", str(report),
            "--skip-ctest", "--skip-catalog", "--timeout", "120",
        )
        data = self.load_report(proc, report)

        self.assertEqual(data["schema"], "host-validation-report/1")
        self.assertEqual(data["overall"]["status"], "passed")
        self.assertEqual(data["overall"]["reasons"], [])
        # Source identity recorded from the fixture git repository.
        self.assertEqual(data["source"]["commit"],
                         subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                                        capture_output=True, text=True,
                                        check=True).stdout.strip())
        self.assertFalse(data["source"]["source_drift"])
        # Content identity: deterministic digest over tracked + relevant
        # untracked files, stable across the run.
        manifest = data["source"]["manifest"]
        self.assertEqual(manifest["digest_start"], manifest["digest_end"])
        self.assertRegex(manifest["digest_start"], r"^[0-9a-f]{64}$")
        self.assertGreater(manifest["file_count_start"], 0)
        self.assertFalse(data["source"]["content_drift"])
        # A dirty checkout passes but carries its provenance.
        self.assertEqual(data["source"]["dirty_count_start"],
                         data["source"]["dirty_count_end"])
        # Artifacts directory next to the report, never inside the source.
        artifacts = out / "host-validation-artifacts"
        self.assertTrue(artifacts.is_dir())
        self.assertTrue(str(artifacts).startswith(str(out)))
        self.assertNotIn(str(repo), str(artifacts))
        # Explicitly skipped sections are visible, not silent.
        self.assertEqual(data["ctest"]["status"], "skipped-explicit")
        self.assertEqual(data["catalog"]["status"], "skipped-explicit")
        self.assertFalse(data["options"]["ci_equivalent"])
        # The options record the reduction of scope.
        self.assertEqual(sorted(data["options"]["skipped_sections"]),
                         ["catalog", "ctest"])
        # The green fixture ships the accepted inventory and the coverage
        # cross-check passes against it.
        self.assertEqual(data["sample_coverage"]["status"], "ok")
        self.assertEqual(data["sample_coverage"]["expected_samples"], 5)

    def test_identical_module_names_isolated_with_real_counts(self):
        repo = self.green_repo()
        report = self.tmp / "iso" / "report.json"
        proc = run_runner(repo, "--report", str(report),
                          "--skip-ctest", "--skip-catalog", "--timeout", "120")
        data = self.load_report(proc, report)

        alpha = self.suite_by_dir(data, "samples/vision/alpha/tests")
        beta = self.suite_by_dir(data, "samples/vision/beta/tests")
        self.assertEqual(alpha["status"], "passed")
        self.assertEqual(beta["status"], "passed")
        self.assertEqual(alpha["tests"], 2)
        self.assertEqual(beta["tests"], 1)

    def test_nested_conversion_torch_import_recorded_as_optional(self):
        repo = self.green_repo()
        report = self.tmp / "optional" / "report.json"
        proc = run_runner(repo, "--report", str(report),
                          "--skip-ctest", "--skip-catalog", "--timeout", "120")
        data = self.load_report(proc, report)

        suite = self.suite_by_dir(data, "samples/vision/gamma/conversion/tests")
        self.assertEqual(suite["status"], "passed")
        self.assertEqual(
            suite["optional_missing"],
            [{"file": "test_export_heads.py", "missing": "torch"}],
        )
        # The importable module in the same directory still ran.
        self.assertEqual(suite["tests"], 1)
        # No error was invented for the optional export scope.
        self.assertEqual(suite["errors"], 0)
        # ... and the optional miss is part of the summary, not hidden.
        self.assertEqual(data["python_suites"]["totals"]
                         ["optional_missing_modules"], 1)

    def test_paraformer_export_stage_skip_is_optional_scope(self):
        repo = self.green_repo()
        report = self.tmp / "paraformer" / "report.json"
        proc = run_runner(repo, "--report", str(report),
                          "--skip-ctest", "--skip-catalog", "--timeout", "120")
        data = self.load_report(proc, report)

        suite = self.suite_by_dir(data, "samples/speech/paraformer/tests")
        self.assertEqual(suite["status"], "passed")
        self.assertEqual(suite["skipped"], 1)
        self.assertEqual(suite["skips"][0]["category"], "optional_export")
        self.assertIn("test_export_stages", suite["skips"][0]["id"])

    def test_vla_parent_guard_runs_in_shared_suite(self):
        repo = self.green_repo()
        report = self.tmp / "vla" / "report.json"
        proc = run_runner(repo, "--report", str(report),
                          "--skip-ctest", "--skip-catalog", "--timeout", "120")
        data = self.load_report(proc, report)

        # The VLA parent-repo guard executed with the rest of the shared
        # suite: both test files ran and nothing is excluded anymore.
        suite = self.suite_by_dir(data, "utils/py_utils/tests")
        self.assertEqual(suite["tests"], 2)
        self.assertEqual(suite["status"], "passed")
        self.assertNotIn("excluded_files", data["python_suites"])

    def test_native_only_directory_recorded_not_run_as_python(self):
        repo = self.green_repo()
        report = self.tmp / "nativeonly" / "report.json"
        proc = run_runner(repo, "--report", str(report),
                          "--skip-ctest", "--skip-catalog", "--timeout", "120")
        data = self.load_report(proc, report)
        self.assertIn("samples/vision/alpha/runtime/cpp/tests",
                      data["python_suites"]["native_only_dirs"])
        dirs = [s["dir"] for s in data["python_suites"]["suites"]]
        self.assertNotIn("samples/vision/alpha/runtime/cpp/tests", dirs)

    def test_contract_section_runs_repo_checker_and_records_summary(self):
        repo = self.green_repo()
        report = self.tmp / "contract" / "report.json"
        proc = run_runner(repo, "--report", str(report),
                          "--skip-ctest", "--skip-catalog", "--timeout", "120")
        data = self.load_report(proc, report)
        contract = data["contract"]
        self.assertEqual(contract["status"], "passed")
        self.assertEqual(contract["exit_code"], 0)
        self.assertEqual(contract["summary"]["violations"], 0)
        self.assertIn("--scope", contract["command"])
        # The checker's own report file is kept outside the source tree.
        self.assertTrue(Path(contract["report_file"]).is_file())
        self.assertNotIn(str(repo), contract["report_file"])


class RequestedRepoCwdTests(RunnerFixtureTestCase):
    """Suite subprocesses execute at the requested repository root.

    Regression for the 2026-10-05 independent clean-clone delivery run:
    launched from outside the clone with an absolute ``--repo`` and a
    cleared ``PYTHONPATH``, the Python suites inherited the caller's
    directory, so README-command tests resolved repo-relative paths
    against the wrong cwd and nested child interpreters could not import
    ``samples`` — an orchestration defect, since the same suites pass when
    the repository root is the working directory.  Suite subprocesses now
    start at the requested repository, deterministically for any
    maintainer launch directory, while subprocess isolation, the
    outside-the-source report rule and the caller's own working directory
    all stay intact.
    """

    def probe_repo(self) -> Path:
        cls = type(self)
        if not hasattr(cls, "_cwd_probe_repo"):
            cls._cwd_probe_repo = _init_repo(self.tmp / "cwd-probe",
                                             _cwd_probe_layout)
        return cls._cwd_probe_repo

    def test_outside_launch_runs_suites_at_requested_repo(self):
        repo = self.probe_repo()
        # A distinct temporary launch directory: outside the fixture
        # repository and different from this process's own cwd.
        launch_dir = self.tmp / "outside-launch"
        launch_dir.mkdir(exist_ok=True)
        caller_cwd = Path.cwd()
        self.assertNotEqual(launch_dir, caller_cwd)
        self.assertNotIn(str(repo), str(launch_dir))
        self.assertNotIn(str(launch_dir), str(repo))
        out = self.tmp / "cwd-reports"
        report = out / "report.json"
        proc = run_runner_outside(
            launch_dir, repo, "--report", str(report),
            "--skip-ctest", "--skip-catalog", "--timeout", "120")
        # The maintainer never relocates its caller.
        self.assertEqual(Path.cwd(), caller_cwd)

        data = self.load_report(proc, report)
        self.assertEqual(data["overall"]["status"], "passed")
        self.assertEqual(data["overall"]["reasons"], [])
        # Exact executed counts: every discovered fixture suite ran.
        self.assertEqual(data["python_suites"]["discovered"], 7)
        self.assertEqual(data["python_suites"]["selected"], 7)
        self.assertEqual(data["python_suites"]["totals"],
                         {"tests": 9, "failures": 0, "errors": 0,
                          "skipped": 0, "optional_missing_modules": 0})
        probe = self.suite_by_dir(data, "samples/vision/cwdprobe/tests")
        self.assertEqual(probe["status"], "passed")
        self.assertEqual(probe["tests"], 3)
        self.assertEqual(probe["failures"], 0)
        self.assertEqual(probe["errors"], 0)
        # The requested repository is unchanged and clean.
        self.assertFalse(data["source"]["source_drift"])
        self.assertFalse(data["source"]["content_drift"])
        self.assertEqual(data["source"]["dirty_files_start"], [])
        self.assertEqual(data["source"]["dirty_files_end"], [])
        porcelain = subprocess.run(
            ["git", "-C", str(repo), "status", "--porcelain"],
            capture_output=True, text=True)
        self.assertEqual(porcelain.returncode, 0)
        self.assertEqual(porcelain.stdout, "")
        # Report and artifacts stay outside the source tree.
        self.assertNotIn(str(repo), str(report))
        artifacts = out / "host-validation-artifacts"
        self.assertTrue(artifacts.is_dir())
        self.assertNotIn(str(repo), str(artifacts))


class FailureModeTests(RunnerFixtureTestCase):
    """Every failure mode reports honestly and blocks a passing exit code."""

    def _build_problem_repo(self) -> Path:
        def layout(repo: Path) -> None:
            _green_layout(repo)
            _write(
                repo / "samples/vision/beta/tests/test_model.py",
                """
                import unittest

                class BetaModelTests(unittest.TestCase):
                    def test_ok(self):
                        self.assertTrue(True)

                    def test_fail(self):
                        self.assertEqual(1, 2)

                    def test_error(self):
                        raise RuntimeError("boom")
                """,
            )
            _write(
                repo / "samples/vision/delta/tests/test_native.py",
                """
                import unittest

                class NativeTests(unittest.TestCase):
                    def test_native(self):
                        self.skipTest("C++17 compiler unavailable; entry not-run")
                """,
            )
            # An UNRELATED runtime skip that names torch/funasr: outside the
            # declared export scope, so it must be rejected, not allowed.
            _write(
                repo / "samples/vision/eps/tests/test_opt.py",
                """
                import unittest

                @unittest.skip("Torch and FunASR export dependencies required")
                class OptionalTests(unittest.TestCase):
                    def test_opt(self):
                        self.assertTrue(True)
                """,
            )
            # A discovered suite whose only test file defines no test cases.
            _write(
                repo / "samples/vision/empty/tests/test_notestcases.py",
                """
                def helper():
                    return 1
                """,
            )
            # A tests directory with neither Python tests nor native markers.
            _write(
                repo / "samples/vision/odd/tests/helper.py",
                "def helper():\n    return 1\n",
            )
        return _init_repo(self.tmp / "problem", layout)

    def test_failures_errors_and_reasons_are_reported(self):
        repo = self.problem_repo()
        report = self.tmp / "problem" / "report.json"
        proc = run_runner(repo, "--report", str(report),
                          "--skip-ctest", "--skip-catalog", "--timeout", "120")
        self.assertEqual(proc.returncode, 1)
        data = json.loads(report.read_text())
        self.assertEqual(data["overall"]["status"], "failed")

        beta = self.suite_by_dir(data, "samples/vision/beta/tests")
        self.assertEqual(beta["status"], "failed")
        self.assertEqual(beta["tests"], 3)
        self.assertEqual(beta["failures"], 1)
        self.assertEqual(beta["errors"], 1)
        failed = {entry["id"] for entry in beta["failures_detail"]}
        errored = {entry["id"] for entry in beta["errors_detail"]}
        self.assertEqual(len(failed), 1)
        self.assertEqual(len(errored), 1)
        self.assertIn("test_fail", " ".join(failed))
        self.assertIn("test_error", " ".join(errored))
        self.assertIn("boom", beta["errors_detail"][0]["message"])

    def test_native_prerequisite_skip_fails_strict_mode(self):
        repo = self.problem_repo()
        report = self.tmp / "native-strict" / "report.json"
        proc = run_runner(repo, "--report", str(report),
                          "--skip-ctest", "--skip-catalog", "--timeout", "120")
        self.assertEqual(proc.returncode, 1)
        data = json.loads(report.read_text())
        reasons = " ".join(data["overall"]["reasons"])
        self.assertIn("native-prerequisite skip", reasons)
        delta = self.suite_by_dir(data, "samples/vision/delta/tests")
        self.assertEqual(delta["skips"][0]["category"], "native_prerequisite")

    def test_allow_native_skips_records_but_does_not_invent_success(self):
        repo = self.problem_repo()
        report = self.tmp / "native-allow" / "report.json"
        proc = run_runner(repo, "--report", str(report), "--allow-native-skips",
                          "--skip-ctest", "--skip-catalog", "--timeout", "120")
        # The repository still has real failures; allowing native skips must
        # not turn the run green -- it only downgrades the skip policy.
        self.assertEqual(proc.returncode, 1)
        data = json.loads(report.read_text())
        reasons = " ".join(data["overall"]["reasons"])
        self.assertNotIn("native-prerequisite skip", reasons)
        self.assertFalse(data["options"]["ci_equivalent"])
        delta = self.suite_by_dir(data, "samples/vision/delta/tests")
        self.assertEqual(delta["status"], "passed")
        self.assertEqual(delta["skips"][0]["category"], "native_prerequisite")
        self.assertEqual(delta["skips"][0]["policy"], "allowed-by-option")

    def test_unrelated_runtime_framework_skip_is_rejected(self):
        repo = self.problem_repo()
        report = self.tmp / "opt-skip" / "report.json"
        proc = run_runner(repo, "--report", str(report),
                          "--skip-ctest", "--skip-catalog", "--timeout", "120",
                          "--suite", "samples/vision/eps/tests")
        self.assertEqual(proc.returncode, 1)
        data = json.loads(report.read_text())
        eps = self.suite_by_dir(data, "samples/vision/eps/tests")
        # The framework-named skip is outside the declared export scope
        # (only conversion-test-dir import misses and the Paraformer
        # export-stage identities are optional): rejected as undeclared.
        self.assertEqual(eps["status"], "failed")
        self.assertEqual(eps["skips"][0]["category"], "unexpected")
        reasons = " ".join(data["overall"]["reasons"])
        self.assertIn("undeclared skip", reasons)

    def test_zero_test_directory_cannot_report_success(self):
        repo = self.problem_repo()
        report = self.tmp / "zero" / "report.json"
        proc = run_runner(repo, "--report", str(report),
                          "--skip-ctest", "--skip-catalog", "--timeout", "120",
                          "--suite", "samples/vision/empty/tests")
        self.assertEqual(proc.returncode, 1)
        data = json.loads(report.read_text())
        empty = self.suite_by_dir(data, "samples/vision/empty/tests")
        self.assertEqual(empty["status"], "zero-tests")
        reasons = " ".join(data["overall"]["reasons"])
        self.assertIn("zero tests", reasons)
        # An unexplained directory (no tests, no native markers) is a
        # discovery anomaly, also rejected.
        self.assertTrue(
            any(item.startswith("samples/vision/odd/tests:")
                for item in data["python_suites"]["anomalies"]),
            data["python_suites"]["anomalies"])
        self.assertIn("unexplained test directory", reasons)

    def test_worker_crash_reported_as_worker_error(self):
        def layout(repo: Path) -> None:
            _green_layout(repo)
            _write(
                repo / "samples/vision/crash/tests/test_crash.py",
                """
                import os
                os._exit(3)
                """,
            )
        repo = _init_repo(self.tmp / "crash", layout)
        report = self.tmp / "crash" / "report.json"
        proc = run_runner(repo, "--report", str(report),
                          "--skip-ctest", "--skip-catalog", "--timeout", "120",
                          "--suite", "samples/vision/crash/tests")
        self.assertEqual(proc.returncode, 1)
        data = json.loads(report.read_text())
        crash = self.suite_by_dir(data, "samples/vision/crash/tests")
        self.assertEqual(crash["status"], "worker-error")
        self.assertEqual(crash["exit_code"], 3)

    def test_loader_crash_is_serialized_honestly(self):
        def layout(repo: Path) -> None:
            _green_layout(repo)
            # A load_tests hook that explodes makes loader.discover itself
            # raise inside the worker; the failure must be serialized as an
            # honest loader error, not crash the worker a second time.
            _write(
                repo / "samples/vision/loaderblow/tests/test_loader.py",
                """
                import unittest

                class LoaderTests(unittest.TestCase):
                    def test_loader(self):
                        self.assertTrue(True)

                def load_tests(loader, tests, pattern):
                    raise KeyboardInterrupt("loader blew up")
                """,
            )
        repo = _init_repo(self.tmp / "loaderblow", layout)
        report = self.tmp / "loaderblow" / "report.json"
        proc = run_runner(repo, "--report", str(report),
                          "--skip-ctest", "--skip-catalog", "--timeout", "120",
                          "--suite", "samples/vision/loaderblow/tests")
        self.assertEqual(proc.returncode, 1)
        data = json.loads(report.read_text())
        suite = self.suite_by_dir(data, "samples/vision/loaderblow/tests")
        self.assertEqual(suite["status"], "failed")
        self.assertTrue(suite["loader_errors"])
        error_text = suite["loader_errors"][0]["error"]
        self.assertIn("KeyboardInterrupt", error_text)
        self.assertIn("loader blew up", error_text)
        self.assertIn("load_tests", error_text)

    def test_suite_timeout_is_bounded_and_reported(self):
        def layout(repo: Path) -> None:
            _green_layout(repo)
            _write(
                repo / "samples/vision/slow/tests/test_slow.py",
                """
                import time, unittest

                class SlowTests(unittest.TestCase):
                    def test_sleeps(self):
                        time.sleep(60)
                """,
            )
        repo = _init_repo(self.tmp / "slow", layout)
        report = self.tmp / "slow" / "report.json"
        started = time.monotonic()
        proc = run_runner(repo, "--report", str(report),
                          "--skip-ctest", "--skip-catalog",
                          "--timeout", "5",
                          "--suite", "samples/vision/slow/tests",
                          timeout=120)
        duration = time.monotonic() - started
        self.assertEqual(proc.returncode, 1)
        self.assertLess(duration, 50, "timeout not enforced")
        data = json.loads(report.read_text())
        slow = self.suite_by_dir(data, "samples/vision/slow/tests")
        self.assertEqual(slow["status"], "timeout")
        reasons = " ".join(data["overall"]["reasons"])
        self.assertIn("timeout", reasons)

    def test_missing_declared_directory_fails(self):
        def layout(repo: Path) -> None:
            _green_layout(repo)
        repo = _init_repo(self.tmp / "missing", layout)
        # Remove a declared tool suite directory after committing.
        shutil.rmtree(repo / "skills/tests")
        _git(repo, "add", "-A")
        _git(repo, "commit", "-q", "-m", "drop skills tests")
        report = self.tmp / "missing" / "report.json"
        proc = run_runner(repo, "--report", str(report),
                          "--skip-ctest", "--skip-catalog", "--timeout", "120")
        self.assertEqual(proc.returncode, 1)
        data = json.loads(report.read_text())
        self.assertIn("skills/tests", data["python_suites"]["missing_directories"])
        reasons = " ".join(data["overall"]["reasons"])
        self.assertIn("missing declared", reasons)

    def test_missing_pinned_commit_fails_explicitly(self):
        def layout(repo: Path) -> None:
            _green_layout(repo)
            _write(
                repo / "utils/py_utils/legacy_platforms.py",
                f'PIN = "{ABSENT_PIN}"\n',
            )
        repo = _init_repo(self.tmp / "pin-missing", layout)
        report = self.tmp / "pin-missing" / "report.json"
        proc = run_runner(repo, "--report", str(report),
                          "--skip-ctest", "--skip-catalog", "--timeout", "120")
        self.assertEqual(proc.returncode, 1)
        data = json.loads(report.read_text())
        pins = data["source"]["pins"]
        self.assertEqual(len(pins), 1)
        self.assertEqual(pins[0]["pin"], ABSENT_PIN)
        self.assertFalse(pins[0]["present"])
        reasons = " ".join(data["overall"]["reasons"])
        self.assertIn(ABSENT_PIN, reasons)
        self.assertIn(f"git fetch origin {ABSENT_PIN}", reasons)

    def test_present_pinned_commit_passes_pin_verification(self):
        repo = self.tmp / "pin-present"
        subprocess.run(["git", "init", "-q", str(repo)], check=True,
                       capture_output=True)
        _green_layout(repo)
        _git(repo, "add", "-A")
        _git(repo, "commit", "-q", "-m", "base")
        head = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True).stdout.strip()
        # Declare the pre-edit commit as the pin, then move HEAD forward so
        # the pin only resolves through Git history, not the worktree.
        _write(repo / "utils/py_utils/legacy_platforms.py",
               f'PIN = "{head}"\n')
        _write(repo / "marker.txt", "x")
        _git(repo, "add", "-A")
        _git(repo, "commit", "-q", "-m", "declare pin")
        report = self.tmp / "pin-present-reports" / "report.json"
        proc = run_runner(repo, "--report", str(report),
                          "--skip-ctest", "--skip-catalog", "--timeout", "120")
        self.assertEqual(proc.returncode, 0)
        data = json.loads(report.read_text())
        self.assertEqual(data["source"]["pins"][0]["pin"], head)
        self.assertTrue(data["source"]["pins"][0]["present"])


class SourceDriftTests(RunnerFixtureTestCase):
    """Content identity: HEAD-only snapshots cannot certify a green run."""

    def _mutator_layout(self, target_relative: str, action: str):
        def layout(repo: Path) -> None:
            _green_layout(repo)
            # The mutator suite adds a sample directory: the fixture
            # inventory must list it so these drift scenarios exercise the
            # content gate, not the coverage gate.
            _write_inventory(repo, GREEN_INVENTORY_SAMPLES + ("vision/zeta",))
            _write(
                repo / "samples/vision/zeta/tests/test_mutator.py",
                f"""
                import pathlib
                import unittest

                TARGET = (pathlib.Path(__file__).resolve().parents[4]
                          / pathlib.PurePosixPath({target_relative!r}))

                class MutatorTests(unittest.TestCase):
                    def test_mutate_target(self):
                        TARGET.parent.mkdir(parents=True, exist_ok=True)
                        if {action!r} == "append":
                            TARGET.write_text(
                                TARGET.read_text() + "mutated\\n")
                        else:  # ignored-write
                            TARGET.write_text("noise")
                        self.assertTrue(True)
                """,
            )
        return layout

    def test_tracked_content_mutation_with_unchanged_dirty_list_fails(self):
        # Pre-dirty the tracked file so its name is already in the dirty
        # list; the mutator then changes its CONTENT during the run.  HEAD
        # and the dirty filename list stay identical.
        repo = _init_repo(
            self.tmp / "drift-tracked",
            self._mutator_layout("samples/vision/zeta/README.md", "append"))
        target = repo / "samples/vision/zeta/README.md"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("initial\ndirty\n")
        before = subprocess.run(
            ["git", "-C", str(repo), "status", "--porcelain"],
            capture_output=True, text=True, check=True).stdout
        report = self.tmp / "drift-tracked" / "report.json"
        proc = run_runner(repo, "--report", str(report),
                          "--skip-ctest", "--skip-catalog", "--timeout", "120")
        after = subprocess.run(
            ["git", "-C", str(repo), "status", "--porcelain"],
            capture_output=True, text=True, check=True).stdout
        self.assertEqual(before, after, "fixture must not change the dirty list")
        self.assertEqual(proc.returncode, 1)
        data = json.loads(report.read_text())
        self.assertTrue(data["source"]["content_drift"])
        self.assertNotEqual(data["source"]["manifest"]["digest_start"],
                            data["source"]["manifest"]["digest_end"])
        reasons = " ".join(data["overall"]["reasons"])
        self.assertIn("source content changed during the run", reasons)
        self.assertEqual(data["source"]["commit"],
                         data["source"]["start_commit"])

    def test_untracked_file_content_mutation_with_unchanged_list_fails(self):
        # An untracked-but-relevant file (not gitignored) whose content
        # changes during the run: same dirty list, different content.
        repo = _init_repo(
            self.tmp / "drift-untracked",
            self._mutator_layout("untracked_note.txt", "append"))
        (repo / "untracked_note.txt").write_text("v1\n")
        report = self.tmp / "drift-untracked" / "report.json"
        proc = run_runner(repo, "--report", str(report),
                          "--skip-ctest", "--skip-catalog", "--timeout", "120")
        self.assertEqual(proc.returncode, 1)
        data = json.loads(report.read_text())
        self.assertTrue(data["source"]["content_drift"])
        reasons = " ".join(data["overall"]["reasons"])
        self.assertIn("source content changed during the run", reasons)

    def test_new_untracked_file_during_run_fails(self):
        def layout(repo: Path) -> None:
            _green_layout(repo)
            _write(
                repo / "samples/vision/newfile/tests/test_newfile.py",
                """
                import pathlib
                import unittest

                class NewFileTests(unittest.TestCase):
                    def test_write_untracked(self):
                        (pathlib.Path(__file__).resolve().parents[4]
                         / "surprise.txt").write_text("appeared\\n")
                        self.assertTrue(True)
                """,
            )
        repo = _init_repo(self.tmp / "drift-newfile", layout)
        report = self.tmp / "drift-newfile" / "report.json"
        proc = run_runner(repo, "--report", str(report),
                          "--skip-ctest", "--skip-catalog", "--timeout", "120")
        self.assertEqual(proc.returncode, 1)
        data = json.loads(report.read_text())
        reasons = " ".join(data["overall"]["reasons"])
        self.assertTrue(
            data["source"]["content_drift"]
            or data["source"]["dirty_files_changed_during_run"],
            "new untracked file must be detected")
        self.assertTrue(
            any("surprise.txt" in line
                for line in (data["source"]["dirty_files_end"] or [])))

    def test_ignored_artifacts_do_not_count_as_drift(self):
        # Writes into gitignored paths (caches, build noise) must not fail
        # the gate: the manifest honors Git ignore rules.  The path
        # contains __pycache__, which .gitignore excludes.
        repo = _init_repo(
            self.tmp / "drift-ignored",
            self._mutator_layout(
                "samples/vision/zeta/__pycache__/cache.bin", "ignored-write"))
        report = self.tmp / "drift-ignored" / "report.json"
        proc = run_runner(repo, "--report", str(report),
                          "--skip-ctest", "--skip-catalog", "--timeout", "120")
        data = self.load_report(proc, report)
        self.assertFalse(data["source"]["content_drift"])
        self.assertFalse(data["source"]["dirty_files_changed_during_run"])

    def test_dirty_checkout_passes_with_recorded_provenance(self):
        repo = self.green_repo()
        (repo / "README_extra.md").write_text("local notes\n")
        report = self.tmp / "dirty-green" / "report.json"
        proc = run_runner(repo, "--report", str(report),
                          "--skip-ctest", "--skip-catalog", "--timeout", "120")
        data = self.load_report(proc, report)
        self.assertEqual(data["overall"]["status"], "passed")
        self.assertIn("?? README_extra.md",
                      data["source"]["dirty_files_start"])
        self.assertEqual(data["source"]["dirty_count_start"], 1)

    def test_gitlink_entries_are_not_hashed(self):
        module = _import_runner()
        repo = _init_repo(
            self.tmp / "gitlink",
            self._mutator_layout("samples/vision/zeta/README.md", "append"))
        # Register a gitlink row exactly the way the VLA pins exist in the
        # real tree: no submodule checkout exists to hash.
        subprocess.run(
            ["git", "-C", str(repo), "update-index", "--add", "--cacheinfo",
             f"160000,{ABSENT_PIN},samples/vla/act"],
            check=True, capture_output=True)
        manifest = module.source_manifest(repo)
        self.assertIsNotNone(manifest)
        self.assertIn("samples/vla/act", manifest["gitlinks"])
        self.assertNotIn("samples/vla/act", manifest["files_hashed_sample"])
        self.assertEqual(
            manifest["digest"], module.source_manifest(repo)["digest"])

    def test_non_git_checkout_cannot_pass(self):
        plain = self.tmp / "not-a-repo"
        _green_layout(plain)
        report = self.tmp / "not-a-repo-reports" / "report.json"
        proc = run_runner(plain, "--report", str(report),
                          "--skip-ctest", "--skip-catalog", "--timeout", "120")
        self.assertEqual(proc.returncode, 1)
        data = json.loads(report.read_text())
        self.assertEqual(data["overall"]["status"], "failed")
        self.assertIsNone(data["source"]["commit"])
        self.assertIsNone(data["source"]["manifest"])
        reasons = " ".join(data["overall"]["reasons"])
        self.assertIn("source identity unavailable", reasons)
        self.assertIn("Git", reasons)

    def test_report_path_inside_source_tree_is_rejected(self):
        repo = self.green_repo()
        report = repo / "host-validation-report" / "report.json"
        proc = run_runner(repo, "--report", str(report),
                          "--skip-ctest", "--skip-catalog", "--timeout", "120")
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("outside", proc.stderr)
        # Nothing was written into the source tree.
        self.assertFalse(report.exists())
        self.assertFalse((repo / "host-validation-report").exists())


class CTestDefaultTests(RunnerFixtureTestCase):
    """Default-on native suites and per-stage failure containment."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.module = _import_runner()

    def _fixture_project_layout(self, repo: Path) -> None:
        _green_layout(repo)
        # A real (tiny) CTest project at the yoloe registry path that
        # verifies the default-on define actually reaches CMake.
        _write(
            repo / "samples/vision/yoloe/runtime/cpp/tests/CMakeLists.txt",
            """
            cmake_minimum_required(VERSION 3.16)
            project(fixture_yoloe_tests LANGUAGES CXX)
            if(NOT DEFINED YOLOE_TEST_OPENCV)
              message(FATAL_ERROR "YOLOE_TEST_OPENCV define missing")
            endif()
            if(NOT YOLOE_TEST_OPENCV STREQUAL "ON")
              message(FATAL_ERROR "YOLOE_TEST_OPENCV must default to ON")
            endif()
            enable_testing()
            add_executable(fixture_ok fixture_ok.cc)
            add_test(NAME fixture_ok COMMAND fixture_ok)
            """,
        )
        _write(
            repo / "samples/vision/yoloe/runtime/cpp/tests/fixture_ok.cc",
            "int main() { return 0; }\n",
        )

    def _registry(self, name="yoloe-cpp-tests"):
        return ({"name": name,
                 "source": "samples/vision/yoloe/runtime/cpp/tests",
                 "note": "fixture"},)

    def _options(self, **overrides):
        options = {
            "suite_filters": [], "timeout_s": 120, "allow_native_skips": False,
            "skip_ctest": False, "skip_catalog": True,
            "skipped_sections": ["catalog"], "cmake": None, "ctest": None,
            "cmake_defines": {},
        }
        options.update(overrides)
        return options

    def test_default_definitions_are_the_full_gate(self):
        self.assertEqual(
            self.module.DEFAULT_CTEST_DEFINES,
            {"yoloe-cpp-tests": {"YOLOE_TEST_OPENCV": "ON"},
             "asr-cpp-tests": {"ASR_AUDIO_TESTS": "ON",
                               "ASR_CLI_TESTS": "ON"},
             "paraformer-cpp-tests": {
                 "PARAFORMER_BUILD_TESTS": "ON",
                 "PARAFORMER_BUILD_IO": "ON",
                 "PARAFORMER_SANITIZERS": "ON",
                 "PARAFORMER_BUILD_SDK": "OFF",
                 "PARAFORMER_BUILD_CLI": "OFF"},
             "himloco-cpp-tests": {
                 "HIMLOCO_BUILD_TESTS": "ON",
                 "HIMLOCO_BUILD_SDK": "OFF",
                 "HIMLOCO_BUILD_CLI": "OFF"}},
        )
        # The vendor/production build switches the gate mandates OFF.  Those
        # OFF defaults are mandatory safety, not a reducible scope choice.
        self.assertEqual(
            self.module.MANDATORY_SAFE_DEFINES,
            {"paraformer-cpp-tests": {"PARAFORMER_BUILD_SDK": "OFF",
                                      "PARAFORMER_BUILD_CLI": "OFF"},
             "himloco-cpp-tests": {"HIMLOCO_BUILD_SDK": "OFF",
                                   "HIMLOCO_BUILD_CLI": "OFF"}})
        registered = {project["name"]
                      for project in self.module.CTEST_PROJECTS}
        for project, defines in self.module.MANDATORY_SAFE_DEFINES.items():
            self.assertIn(project, registered,
                          "mandatory-safe project not registered")
            for key, value in defines.items():
                self.assertEqual(
                    self.module.DEFAULT_CTEST_DEFINES[project][key], value,
                    f"{project}:{key} must be part of the full-gate defaults")
        for project in self.module.CTEST_PROJECTS:
            if project.get("requires_opencv"):
                self.assertIn(
                    project["name"],
                    ("gemma4-e2b-native", "yoloe-cpp-tests"),
                    "unknown OpenCV-requiring project")

    @unittest.skipUnless(_interpreter_tool("cmake") and _interpreter_tool("ctest"),
                         "native prerequisite (cmake) unavailable")
    def test_full_gate_project_receives_default_on_defines(self):
        repo = _init_repo(self.tmp / "ctest-defaults",
                          self._fixture_project_layout)
        original = self.module.CTEST_PROJECTS
        self.module.CTEST_PROJECTS = self._registry()
        try:
            runner = self.module._SectionRunner(
                repo, self.tmp / "ctest-defaults" / "report.json",
                sys.executable, self._options())
            section = runner.run_ctest()
        finally:
            self.module.CTEST_PROJECTS = original
        self.assertEqual(section["status"], "passed", section)
        project = section["projects"][0]
        self.assertEqual(project["defines"], {"YOLOE_TEST_OPENCV": "ON"})
        self.assertEqual(project["total"], 1)
        self.assertEqual(project["declared_tests"], 1)

    def _guarded_project_layout(self, source_rel: str, project: str):
        """A guard CMakeLists at a new registry path.

        It fails at configure time unless the exact full-gate safe defaults
        arrive on the command line, so the fixture proves the runner passes
        the declared flags (including the mandated OFF vendor switches), not
        merely that a project at that path configures.
        """
        defines = self.module.DEFAULT_CTEST_DEFINES[project]
        checks = "\n".join(
            f'            "{key}={value}"' for key, value in defines.items())

        def layout(repo: Path) -> None:
            _green_layout(repo)
            _write(
                repo / source_rel / "CMakeLists.txt",
                f"""
                cmake_minimum_required(VERSION 3.16)
                project(fixture_guard_tests LANGUAGES CXX)
                foreach(pair IN ITEMS
                {checks}
                )
                  string(REPLACE "=" ";" kv "${{pair}}")
                  list(GET kv 0 name)
                  list(GET kv 1 expected)
                  if(NOT DEFINED ${{name}})
                    message(FATAL_ERROR "define missing: ${{name}}")
                  endif()
                  if(NOT ${{${{name}}}} STREQUAL "${{expected}}")
                    message(FATAL_ERROR "${{name}} must be ${{expected}}")
                  endif()
                endforeach()
                enable_testing()
                add_executable(fixture_ok fixture_ok.cc)
                add_test(NAME fixture_ok COMMAND fixture_ok)
                """,
            )
            _write(
                repo / source_rel / "fixture_ok.cc",
                "int main() { return 0; }\n",
            )
        return layout

    @unittest.skipUnless(_interpreter_tool("cmake") and _interpreter_tool("ctest"),
                         "native prerequisite (cmake) unavailable")
    def test_paraformer_project_receives_safe_defaults(self):
        repo = _init_repo(
            self.tmp / "ctest-paraformer-defaults",
            self._guarded_project_layout(
                "samples/speech/paraformer/runtime/cpp",
                "paraformer-cpp-tests"))
        original = self.module.CTEST_PROJECTS
        self.module.CTEST_PROJECTS = (
            {"name": "paraformer-cpp-tests",
             "source": "samples/speech/paraformer/runtime/cpp",
             "note": "fixture"},)
        try:
            runner = self.module._SectionRunner(
                repo, self.tmp / "ctest-paraformer-defaults" / "report.json",
                sys.executable, self._options())
            section = runner.run_ctest()
        finally:
            self.module.CTEST_PROJECTS = original
        self.assertEqual(section["status"], "passed", section)
        project = section["projects"][0]
        # The exact default set — host tests/IO/sanitizers ON, vendor SDK and
        # production CLI OFF — reaches CMake.
        self.assertEqual(
            project["defines"],
            {"PARAFORMER_BUILD_TESTS": "ON", "PARAFORMER_BUILD_IO": "ON",
             "PARAFORMER_SANITIZERS": "ON", "PARAFORMER_BUILD_SDK": "OFF",
             "PARAFORMER_BUILD_CLI": "OFF"})
        self.assertEqual(project["total"], 1)
        self.assertEqual(project["declared_tests"], 1)

    @unittest.skipUnless(_interpreter_tool("cmake") and _interpreter_tool("ctest"),
                         "native prerequisite (cmake) unavailable")
    def test_himloco_project_receives_safe_defaults(self):
        repo = _init_repo(
            self.tmp / "ctest-himloco-defaults",
            self._guarded_project_layout(
                "samples/robotics/himloco/runtime/cpp",
                "himloco-cpp-tests"))
        original = self.module.CTEST_PROJECTS
        self.module.CTEST_PROJECTS = (
            {"name": "himloco-cpp-tests",
             "source": "samples/robotics/himloco/runtime/cpp",
             "note": "fixture"},)
        try:
            runner = self.module._SectionRunner(
                repo, self.tmp / "ctest-himloco-defaults" / "report.json",
                sys.executable, self._options())
            section = runner.run_ctest()
        finally:
            self.module.CTEST_PROJECTS = original
        self.assertEqual(section["status"], "passed", section)
        project = section["projects"][0]
        self.assertEqual(
            project["defines"],
            {"HIMLOCO_BUILD_TESTS": "ON", "HIMLOCO_BUILD_SDK": "OFF",
             "HIMLOCO_BUILD_CLI": "OFF"})
        self.assertEqual(project["total"], 1)

    def test_disabling_a_default_on_flag_is_not_ci_equivalent(self):
        repo = self.green_repo()
        report = self.tmp / "ctest-off" / "report.json"
        proc = run_runner(repo, "--report", str(report),
                          "--skip-ctest", "--skip-catalog", "--timeout", "120",
                          "--cmake-define",
                          "yoloe-cpp-tests:YOLOE_TEST_OPENCV=OFF")
        data = self.load_report(proc, report)
        self.assertFalse(data["options"]["ci_equivalent"])
        self.assertIn("yoloe-cpp-tests:YOLOE_TEST_OPENCV=OFF",
                      data["options"]["cmake_scope_reductions"])

    def test_disabling_new_project_test_flags_reduces_scope(self):
        # Turning the required host test/IO/sanitizer flags of the added
        # projects OFF is a recorded scope reduction, never full CI
        # equivalence.
        repo = self.green_repo()
        for define in ("paraformer-cpp-tests:PARAFORMER_SANITIZERS=OFF",
                       "paraformer-cpp-tests:PARAFORMER_BUILD_IO=OFF",
                       "himloco-cpp-tests:HIMLOCO_BUILD_TESTS=OFF"):
            with self.subTest(define=define):
                report = self.tmp / "ctest-new-off" / "report.json"
                proc = run_runner(
                    repo, "--report", str(report),
                    "--skip-ctest", "--skip-catalog", "--timeout", "120",
                    "--cmake-define", define)
                data = self.load_report(proc, report)
                self.assertFalse(data["options"]["ci_equivalent"])
                self.assertIn(define, data["options"]["cmake_scope_reductions"])

    def test_cmake_false_constant_aliases_disable_required_flags(self):
        # Real CMake 4.4.4 evaluates the raw -D values "", N, IGNORE,
        # NOTFOUND and missing-NOTFOUND to OFF through option()+if()
        # (independent-cmake-false-constants evidence; the padded " OFF "
        # control there stays ON, and trailing whitespace is stripped by
        # CMake's -D caching itself — see host_cmake_false_scope/execution/
        # extra-cmake-aliases).  Each is a legitimate untyped override that
        # turns a required default-ON test/IO/sanitizer flag off in fact,
        # so the CLI/report path must record it as a scope reduction with
        # the raw value and mark the run non-CI-equivalent — while still
        # emitting the define raw to the configure command.  The fake
        # cmake/ctest sentinels log every invocation (and fake-discover one
        # passing case per project), so no vendor build ever runs and the
        # emitted command line is provable from the log.
        def layout(repo: Path) -> None:
            _green_layout(repo)
            # Every registered project present as a stub, so the run is a
            # complete green checkout whose only variable is the
            # --cmake-define value spelling.
            for project in self.module.CTEST_PROJECTS:
                _write(repo / project["source"] / "CMakeLists.txt",
                       "# fixture stub (never really configured)\n")
                sample = "/".join(project["source"].split("/")[1:3])
                if sample not in GREEN_INVENTORY_SAMPLES:
                    _write(repo / "samples" / sample / "tests" /
                           "CMakeLists.txt",
                           "# native-only marker (fixture stub)\n")
            _write_inventory(
                repo, sorted(set(GREEN_INVENTORY_SAMPLES).union(
                    "/".join(project["source"].split("/")[1:3])
                    for project in self.module.CTEST_PROJECTS)))

        repo = _init_repo(self.tmp / "false-constants", layout)
        logs = self.tmp / "false-constants" / "invocations"
        variants = (
            "paraformer-cpp-tests:PARAFORMER_SANITIZERS=N",
            "paraformer-cpp-tests:PARAFORMER_BUILD_IO=n",
            "paraformer-cpp-tests:PARAFORMER_BUILD_TESTS=IGNORE",
            "himloco-cpp-tests:HIMLOCO_BUILD_TESTS=NOTFOUND",
            "yoloe-cpp-tests:YOLOE_TEST_OPENCV=missing-NOTFOUND",
            "asr-cpp-tests:ASR_AUDIO_TESTS=",
            # Trailing whitespace is stripped by CMake's own -D caching,
            # so this spelling really disables the IO flag: a reduction
            # too, recorded with the raw trailing-space value.
            "paraformer-cpp-tests:PARAFORMER_BUILD_IO=NO ",
        )
        ctest_body = (
            'case "$*" in\n'
            '  *json-v1*) printf \'{"tests":[{"name":"fixture_ok"}]}\\n\' ;;\n'
            "  *) printf '100%% tests passed, 0 tests failed out of 1\\n' ;;\n"
            "esac\n"
            "exit 0")
        for index, define in enumerate(variants):
            with self.subTest(define=define):
                stem = f"false-const-{index}"
                cmake_log = logs / f"{stem}-cmake.log"
                cmake, ctest = self._fake_tools(
                    stem,
                    'echo "$@" >> ' + str(cmake_log) + "\nexit 0",
                    ctest_body)
                report = logs / f"{stem}-report.json"
                proc = run_runner(
                    repo, "--report", str(report),
                    "--skip-catalog", "--cmake", cmake, "--ctest", ctest,
                    "--timeout", "120",
                    "--cmake-define", define)
                data = self.load_report(proc, report)
                self.assertFalse(data["options"]["ci_equivalent"])
                self.assertIn(define,
                              data["options"]["cmake_scope_reductions"])
                self.assertEqual(data["ctest"]["status"], "passed")
                # The reduction is reported, never silently repaired: the
                # raw value reached the configure command line as given.
                emitted = "-D" + define.split(":", 1)[1]
                self.assertIn(emitted, cmake_log.read_text())
        # A symmetrically padded false-looking value keeps the flag ON in
        # fact (CMake preserves the leading whitespace of a -D value), so
        # the same CLI path records no reduction for it and the run stays
        # a plain green pass.
        stem = "false-const-padded"
        cmake_log = logs / f"{stem}-cmake.log"
        cmake, ctest = self._fake_tools(
            stem,
            'echo "$@" >> ' + str(cmake_log) + "\nexit 0",
            ctest_body)
        report = logs / f"{stem}-report.json"
        proc = run_runner(
            repo, "--report", str(report),
            "--skip-catalog", "--cmake", cmake, "--ctest", ctest,
            "--timeout", "120",
            "--cmake-define", "paraformer-cpp-tests:PARAFORMER_SANITIZERS= N ")
        data = self.load_report(proc, report)
        self.assertEqual(data["options"]["cmake_scope_reductions"], [])
        self.assertIn("-DPARAFORMER_SANITIZERS= N ", cmake_log.read_text())

    def test_sdk_or_cli_enabling_overrides_are_rejected_before_build(self):
        # The vendor SDK / production CLI switches are mandated OFF: this
        # maintainer tool never constructs those builds, so an enabling
        # override is rejected outright — before any project is configured
        # and with a reason naming the rejected switch.
        repo = self.green_repo()
        for define in ("paraformer-cpp-tests:PARAFORMER_BUILD_SDK=ON",
                       "paraformer-cpp-tests:PARAFORMER_BUILD_CLI=1",
                       "himloco-cpp-tests:HIMLOCO_BUILD_SDK=ON",
                       "himloco-cpp-tests:HIMLOCO_BUILD_CLI=true"):
            with self.subTest(define=define):
                report = self.tmp / "ctest-protect" / "report.json"
                proc = run_runner(
                    repo, "--report", str(report),
                    "--skip-catalog", "--timeout", "120",
                    "--cmake-define", define)
                self.assertEqual(proc.returncode, 2,
                                 f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")
                variable = define.split(":", 1)[1].split("=", 1)[0]
                self.assertIn(variable, proc.stderr)
                self.assertIn("vendor", proc.stderr.lower())
                # Rejected before any build or report exists.
                self.assertFalse(report.exists())

    def test_restatement_of_safe_off_define_is_accepted_noop(self):
        # Explicitly restating the mandated OFF value is a no-op: not a
        # rejection and not a scope reduction.
        repo = self.green_repo()
        report = self.tmp / "ctest-protect-off" / "report.json"
        proc = run_runner(
            repo, "--report", str(report),
            "--skip-ctest", "--skip-catalog", "--timeout", "120",
            "--cmake-define", "paraformer-cpp-tests:PARAFORMER_BUILD_SDK=OFF")
        data = self.load_report(proc, report)
        self.assertEqual(data["options"]["cmake_scope_reductions"], [])

    def test_typed_define_keys_are_rejected_before_any_configure(self):
        # Typed CMake cache keys (VAR:BOOL, VAR:STRING, ...) are outside the
        # advertised PROJECT:VAR=VALUE grammar, and they are exactly how a
        # mandated-OFF vendor switch could be re-enabled past the exact-key
        # guard (CMake lets a later typed define override the bare value) or
        # a required default-ON flag quietly turned off with no recorded
        # scope reduction.  Every typed spelling is therefore rejected at
        # option parsing — no report is written and no cmake/ctest process
        # ever starts.  The fake tools log each invocation, so an absent log
        # proves rejection happened before the configure stage.
        def layout(repo: Path) -> None:
            _green_layout(repo)
            # Every registered project present, so a define that slipped
            # past would reach the (fake) configure command for all of them;
            # each added sample also carries a marker `tests` directory and
            # an inventory row, so the fixture repo stays a complete green
            # checkout whose only variable is the --cmake-define spelling.
            for project in self.module.CTEST_PROJECTS:
                _write(repo / project["source"] / "CMakeLists.txt",
                       "# fixture stub (never really configured)\n")
                sample = "/".join(project["source"].split("/")[1:3])
                if sample not in GREEN_INVENTORY_SAMPLES:
                    _write(repo / "samples" / sample / "tests" /
                           "CMakeLists.txt",
                           "# native-only marker (fixture stub)\n")
            _write_inventory(
                repo, sorted(set(GREEN_INVENTORY_SAMPLES).union(
                    "/".join(project["source"].split("/")[1:3])
                    for project in self.module.CTEST_PROJECTS)))

        repo = _init_repo(self.tmp / "typed-define", layout)
        logs = self.tmp / "typed-define" / "invocations"
        variants = (
            # Typed enablings of the mandated-OFF vendor/production switches.
            "paraformer-cpp-tests:PARAFORMER_BUILD_SDK:BOOL=ON",
            "paraformer-cpp-tests:PARAFORMER_BUILD_CLI:BOOL=ON",
            "himloco-cpp-tests:HIMLOCO_BUILD_SDK:BOOL=ON",
            "himloco-cpp-tests:HIMLOCO_BUILD_CLI:BOOL=1",
            # Typed disablings of required default-ON test/IO/sanitizer
            # flags — previously escaped scope classification entirely.
            "paraformer-cpp-tests:PARAFORMER_BUILD_TESTS:BOOL=OFF",
            "paraformer-cpp-tests:PARAFORMER_BUILD_IO:BOOL=OFF",
            "paraformer-cpp-tests:PARAFORMER_SANITIZERS:BOOL=OFF",
            "himloco-cpp-tests:HIMLOCO_BUILD_TESTS:BOOL=OFF",
            "yoloe-cpp-tests:YOLOE_TEST_OPENCV:BOOL=OFF",
            # Typed suffixes beyond BOOL and beyond guarded switches.
            "gemma4-e2b-native:GEMMA_PIN:STRING=deadbeef",
            "asr-cpp-tests:ASR_FRONTEND:PATH=/opt/local",
        )
        ctest_body = (
            'echo "$@" >> {ctest_log}\n'
            'case "$*" in\n'
            '  *json-v1*) printf \'{{"tests":[{{"name":"fixture_ok"}}]}}\\n\' ;;\n'
            "  *) printf '100%% tests passed, 0 tests failed out of 1\\n' ;;\n"
            "esac\n"
            "exit 0")
        for index, define in enumerate(variants):
            with self.subTest(define=define):
                stem = f"variant-{index}"
                cmake_log = logs / f"{stem}-cmake.log"
                ctest_log = logs / f"{stem}-ctest.log"
                cmake, ctest = self._fake_tools(
                    stem,
                    'echo "$@" >> ' + str(cmake_log) + "\nexit 0",
                    ctest_body.format(ctest_log=ctest_log))
                report = logs / f"{stem}-report.json"
                proc = run_runner(
                    repo, "--report", str(report),
                    "--skip-catalog", "--cmake", cmake, "--ctest", ctest,
                    "--cmake-define", define)
                evidence = ""
                if cmake_log.exists():
                    evidence = ("\ncmake invocations:\n"
                                + cmake_log.read_text())
                self.assertEqual(
                    proc.returncode, 1,
                    f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
                    f"{evidence}")
                self.assertIn(define, proc.stderr)
                self.assertIn("PROJECT:VAR=VALUE", proc.stderr)
                self.assertIn("typed", proc.stderr.lower())
                # Nothing was configured, tested or reported.
                self.assertFalse(report.exists(), "report must not exist")
                self.assertFalse(cmake_log.exists(),
                                 "cmake must not run before the rejection")
                self.assertFalse(ctest_log.exists(),
                                 "ctest must not run before the rejection")
        # The supported untyped spelling of a vendor switch keeps its own
        # guard rejection (distinct, vendor-named reason) — equally before
        # any configure.
        stem = "untyped-vendor-on"
        cmake_log = logs / f"{stem}-cmake.log"
        ctest_log = logs / f"{stem}-ctest.log"
        cmake, ctest = self._fake_tools(
            stem,
            'echo "$@" >> ' + str(cmake_log) + "\nexit 0",
            'echo "$@" >> ' + str(ctest_log) + "\nexit 0")
        proc = run_runner(
            repo, "--report", str(logs / f"{stem}-report.json"),
            "--skip-catalog", "--cmake", cmake, "--ctest", ctest,
            "--cmake-define", "paraformer-cpp-tests:PARAFORMER_BUILD_SDK=ON")
        self.assertEqual(
            proc.returncode, 2,
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")
        self.assertIn("PARAFORMER_BUILD_SDK", proc.stderr)
        self.assertIn("vendor", proc.stderr.lower())
        self.assertNotIn("typed", proc.stderr.lower())
        self.assertFalse(cmake_log.exists())
        self.assertFalse(ctest_log.exists())

    def test_padded_false_define_tokens_are_rejected_before_any_configure(self):
        # CMake preserves the leading whitespace of a -D value (it strips
        # only trailing whitespace when caching), so a padded false token
        # (" OFF ", " false ", " 0 ") is NOT a trustworthy OFF — with any
        # leading padding the switch is enabled in fact (verified against
        # real CMake 4.4.4 with an option(... OFF) + if(...) probe; see
        # independent-cmake-boolean-whitespace).  The guard accepts only
        # exact, unpadded false-constant spellings as no-op restatements
        # and rejects every padded spelling for the mandated-OFF
        # vendor/production switches fail-closed — never guessing which
        # padding CMake might strip — before any project is configured,
        # with no report written and no cmake/ctest process started.  The
        # fake tools log each invocation, so absent logs prove rejection
        # happened before the configure stage.
        def layout(repo: Path) -> None:
            _green_layout(repo)
            # Every registered project present, so a define that slipped
            # past would reach the (fake) configure command for all of them
            # — the fixture stays a complete green checkout whose only
            # variable is the --cmake-define value spelling.
            for project in self.module.CTEST_PROJECTS:
                _write(repo / project["source"] / "CMakeLists.txt",
                       "# fixture stub (never really configured)\n")
                sample = "/".join(project["source"].split("/")[1:3])
                if sample not in GREEN_INVENTORY_SAMPLES:
                    _write(repo / "samples" / sample / "tests" /
                           "CMakeLists.txt",
                           "# native-only marker (fixture stub)\n")
            _write_inventory(
                repo, sorted(set(GREEN_INVENTORY_SAMPLES).union(
                    "/".join(project["source"].split("/")[1:3])
                    for project in self.module.CTEST_PROJECTS)))

        repo = _init_repo(self.tmp / "padded-false", layout)
        logs = self.tmp / "padded-false" / "invocations"
        variants = [
            # All four mandated-OFF vendor/production switches, each with a
            # surrounded false token CMake would evaluate as enabled.
            ("paraformer-cpp-tests", "PARAFORMER_BUILD_SDK", " OFF "),
            ("paraformer-cpp-tests", "PARAFORMER_BUILD_CLI", " OFF "),
            ("himloco-cpp-tests", "HIMLOCO_BUILD_SDK", " OFF "),
            ("himloco-cpp-tests", "HIMLOCO_BUILD_CLI", " OFF "),
            ("paraformer-cpp-tests", "PARAFORMER_BUILD_SDK", " false "),
            ("paraformer-cpp-tests", "PARAFORMER_BUILD_CLI", " false "),
            ("himloco-cpp-tests", "HIMLOCO_BUILD_SDK", " false "),
            ("himloco-cpp-tests", "HIMLOCO_BUILD_CLI", " false "),
            ("paraformer-cpp-tests", "PARAFORMER_BUILD_SDK", " 0 "),
            ("paraformer-cpp-tests", "PARAFORMER_BUILD_CLI", " 0 "),
            ("himloco-cpp-tests", "HIMLOCO_BUILD_SDK", " 0 "),
            ("himloco-cpp-tests", "HIMLOCO_BUILD_CLI", " 0 "),
            # One-sided padding and the NO spelling fail the same way.
            ("paraformer-cpp-tests", "PARAFORMER_BUILD_SDK", "OFF "),
            ("himloco-cpp-tests", "HIMLOCO_BUILD_CLI", "\tno"),
        ]
        for index, (project, key, value) in enumerate(variants):
            with self.subTest(define=f"{project}:{key}={value!r}"):
                stem = f"padded-{index}"
                cmake_log = logs / f"{stem}-cmake.log"
                ctest_log = logs / f"{stem}-ctest.log"
                cmake, ctest = self._fake_tools(
                    stem,
                    'echo "$@" >> ' + str(cmake_log) + "\nexit 0",
                    'echo "$@" >> ' + str(ctest_log) + "\nexit 0")
                report = logs / f"{stem}-report.json"
                proc = run_runner(
                    repo, "--report", str(report),
                    "--skip-catalog", "--cmake", cmake, "--ctest", ctest,
                    "--cmake-define", f"{project}:{key}={value}")
                evidence = ""
                if cmake_log.exists():
                    evidence = ("\ncmake invocations:\n"
                                + cmake_log.read_text())
                self.assertEqual(
                    proc.returncode, 2,
                    f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
                    f"{evidence}")
                self.assertIn(f"{project}:{key}={value}", proc.stderr)
                self.assertIn("vendor", proc.stderr.lower())
                # Nothing was configured, tested or reported — the padded
                # token never reached a cmake command line.
                self.assertFalse(report.exists(), "report must not exist")
                self.assertFalse(cmake_log.exists(),
                                 "cmake must not run before the rejection")
                self.assertFalse(ctest_log.exists(),
                                 "ctest must not run before the rejection")
        # Fail-closed is scoped to the mandated-OFF switches: benign defines
        # on the same command line — a padded custom string value and a
        # padded false-looking value of a mere default-ON flag — do not
        # become rejection reasons; the vendor switch still rejects the
        # whole run before any configure.
        stem = "padded-benign-mix"
        cmake_log = logs / f"{stem}-cmake.log"
        ctest_log = logs / f"{stem}-ctest.log"
        cmake, ctest = self._fake_tools(
            stem,
            'echo "$@" >> ' + str(cmake_log) + "\nexit 0",
            'echo "$@" >> ' + str(ctest_log) + "\nexit 0")
        proc = run_runner(
            repo, "--report", str(logs / f"{stem}-report.json"),
            "--skip-catalog", "--cmake", cmake, "--ctest", ctest,
            "--cmake-define", "gemma4-e2b-native:SOME_FLAG= padded value ",
            "--cmake-define", "yoloe-cpp-tests:YOLOE_TEST_OPENCV= OFF ",
            "--cmake-define", "himloco-cpp-tests:HIMLOCO_BUILD_CLI= OFF ")
        self.assertEqual(
            proc.returncode, 2,
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")
        self.assertIn("himloco-cpp-tests:HIMLOCO_BUILD_CLI= OFF ",
                      proc.stderr)
        self.assertNotIn("SOME_FLAG", proc.stderr)
        self.assertNotIn("YOLOE_TEST_OPENCV", proc.stderr)
        self.assertFalse(cmake_log.exists())
        self.assertFalse(ctest_log.exists())

    def test_ctest_missing_projects_fail_but_report_exists(self):
        repo = self.green_repo()
        report = self.tmp / "ctest-none" / "report.json"
        proc = run_runner(repo, "--report", str(report),
                          "--skip-catalog", "--timeout", "120")
        self.assertEqual(proc.returncode, 1)
        data = json.loads(report.read_text())
        self.assertIn(data["ctest"]["status"],
                      ("failed", "missing-prerequisite"))
        self.assertTrue(report.is_file(), "report must exist on ctest failure")

    def _fake_tools(self, name: str, cmake_body: str, ctest_body: str):
        directory = self.tmp / "fake-tools" / name
        directory.mkdir(parents=True, exist_ok=True)
        cmake = directory / "cmake"
        cmake.write_text(f"#!/bin/sh\n{cmake_body}\n")
        ctest = directory / "ctest"
        ctest.write_text(f"#!/bin/sh\n{ctest_body}\n")
        cmake.chmod(0o755)
        ctest.chmod(0o755)
        return str(cmake), str(ctest)

    def _run_section(self, repo_name, options):
        """Build a fixture repo plus callables that run the ctest section
        with the registry narrowed to the fixture project (restored after)."""
        repo = _init_repo(self.tmp / repo_name, self._fixture_project_layout)
        original = self.module.CTEST_PROJECTS
        self.module.CTEST_PROJECTS = self._registry()
        runner = self.module._SectionRunner(
            repo, self.tmp / repo_name / "report.json",
            sys.executable, options)

        def run():
            try:
                return runner.run_ctest()
            finally:
                self.module.CTEST_PROJECTS = original
        return repo, run, lambda: list(runner.reasons)

    def test_missing_cmake_executable_is_contained_per_project(self):
        # Two registered projects: both fail to configure because the cmake
        # executable does not exist, but each failure is contained — the
        # section completes, reports both projects and never raises.
        repo = _init_repo(self.tmp / "ctest-missing-exe",
                          self._fixture_project_layout)
        asr_dir = repo / "samples/speech/asr/runtime/cpp/tests"
        asr_dir.mkdir(parents=True, exist_ok=True)
        _write(asr_dir / "CMakeLists.txt", "# placeholder\n")
        original = self.module.CTEST_PROJECTS
        self.module.CTEST_PROJECTS = (self._registry()
                                      + self._registry("asr-cpp-tests"))
        try:
            runner = self.module._SectionRunner(
                repo, self.tmp / "ctest-missing-exe" / "report.json",
                sys.executable,
                self._options(cmake=str(self.tmp / "no-such" / "cmake"),
                              ctest="/bin/true"))
            section = runner.run_ctest()
        finally:
            self.module.CTEST_PROJECTS = original
        self.assertEqual(section["status"], "failed")
        self.assertEqual(len(section["projects"]), 2)
        self.assertTrue(
            all(p["status"] == "configure-error"
                for p in section["projects"]), section)
        self.assertIn("configure", " ".join(runner.reasons))

    def test_configure_failure_retains_output_and_next_project_continues(self):
        cmake, ctest = self._fake_tools(
            "cfgfail",
            'case "$*" in *asr*) exit 0;; '
            '*) echo "boom: opencv probe failed" >&2; exit 7;; esac',
            'if echo "$@" | grep -q show-only; then'
            ' echo \'{"tests": [1]}\'; else'
            ' echo "100% tests passed, 0 tests failed out of 1"; fi')
        repo = _init_repo(self.tmp / "ctest-cfgfail",
                          self._fixture_project_layout)
        asr_dir = repo / "samples/speech/asr/runtime/cpp/tests"
        asr_dir.mkdir(parents=True, exist_ok=True)
        _write(asr_dir / "CMakeLists.txt", "# placeholder\n")
        original = self.module.CTEST_PROJECTS
        self.module.CTEST_PROJECTS = (self._registry()
                                      + self._registry("asr-cpp-tests"))
        try:
            runner = self.module._SectionRunner(
                repo, self.tmp / "ctest-cfgfail" / "report.json",
                sys.executable, self._options(cmake=cmake, ctest=ctest))
            section = runner.run_ctest()
        finally:
            self.module.CTEST_PROJECTS = original
        statuses = {p["name"]: p["status"] for p in section["projects"]}
        self.assertEqual(statuses["yoloe-cpp-tests"], "configure-failed")
        self.assertEqual(statuses["asr-cpp-tests"], "passed")
        configure_log = (self.tmp / "ctest-cfgfail" / "report.json").parent / \
            "host-validation-artifacts" / "logs" / \
            "ctest-yoloe-cpp-tests-configure.log"
        self.assertTrue(configure_log.is_file())
        self.assertIn("boom", configure_log.read_text())

    def test_stage_timeout_is_contained_and_reported(self):
        _, ctest = self._fake_tools(
            "timeout", "exit 0",
            'if echo "$@" | grep -q show-only; then'
            ' echo \'{"tests": [1]}\'; else sleep 120; fi')
        started = time.monotonic()
        repo, run, reasons = self._run_section(
            "ctest-timeout", self._options(
                cmake=shutil.which("true"), ctest=ctest, ctest_stage_timeout_s=3))
        section = run()
        self.assertLess(time.monotonic() - started, 60)
        project = section["projects"][0]
        self.assertEqual(project["status"], "test-timeout")
        self.assertEqual(section["status"], "failed")
        self.assertIn("timeout", " ".join(reasons()))

    def test_invalid_discovery_json_cannot_pass(self):
        _, ctest = self._fake_tools(
            "invalid-json", "exit 0",
            'if echo "$@" | grep -q show-only; then echo "not json"; else'
            ' echo "100% tests passed, 0 tests failed out of 1"; fi')
        repo, run, _ = self._run_section(
            "ctest-bad-json", self._options(cmake=shutil.which("true"), ctest=ctest))
        section = run()
        project = section["projects"][0]
        self.assertEqual(project["status"], "discovery-failed")
        self.assertEqual(section["status"], "failed")

    def test_declared_count_mismatch_cannot_pass(self):
        _, ctest = self._fake_tools(
            "mismatch", "exit 0",
            'if echo "$@" | grep -q show-only; then'
            ' echo \'{"tests": [1, 2, 3]}\'; else'
            ' echo "100% tests passed, 0 tests failed out of 2"; fi')
        repo, run, _ = self._run_section(
            "ctest-mismatch", self._options(cmake=shutil.which("true"), ctest=ctest))
        section = run()
        project = section["projects"][0]
        self.assertEqual(project["status"], "count-mismatch")
        self.assertEqual(project["declared_tests"], 3)
        self.assertEqual(project["total"], 2)
        self.assertEqual(section["status"], "failed")

    def test_zero_declared_tests_cannot_pass(self):
        _, ctest = self._fake_tools(
            "zero", "exit 0",
            'if echo "$@" | grep -q show-only; then echo \'{"tests": []}\';'
            ' else echo "100% tests passed out of 0"; fi')
        repo, run, _ = self._run_section(
            "ctest-zero", self._options(cmake=shutil.which("true"), ctest=ctest))
        section = run()
        project = section["projects"][0]
        self.assertEqual(project["status"], "zero-tests")
        self.assertEqual(section["status"], "failed")


class CatalogEnginesTests(RunnerFixtureTestCase):
    """The declared Node range is enforced, not merely recorded."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.node_version = subprocess.run(
            ["node", "--version"], capture_output=True, text=True)
        cls.npm_path = shutil.which("npm")

    def setUp(self):
        if self.npm_path is None or self.node_version.returncode != 0:
            self.skipTest("native prerequisite (node/npm) unavailable")

    def _catalog_repo(self, name: str, package_json: str) -> Path:
        def layout(repo: Path) -> None:
            _green_layout(repo)
            package = repo / "scripts/tools/catalog-publisher"
            (package / "node_modules").mkdir(parents=True, exist_ok=True)
            package.joinpath("package.json").write_text(package_json + "\n")
        return _init_repo(self.tmp / name, layout)

    def test_satisfied_engines_run_the_check(self):
        version = self.node_version.stdout.strip().lstrip("v")
        major, minor = version.split(".")[:2]
        repo = self._catalog_repo(
            "catalog-ok",
            f'{{"name": "fixture-catalog", "private": true,'
            f'"engines": {{"node": ">={major}.{minor} <99"}},'
            f'"scripts": {{"check": "node -e \\"process.exit(0)\\""}}}}')
        report = self.tmp / "catalog-ok-reports" / "report.json"
        proc = run_runner(repo, "--report", str(report),
                          "--skip-ctest", "--timeout", "120")
        data = self.load_report(proc, report)
        section = data["catalog"]
        self.assertEqual(section["status"], "passed", section)
        self.assertTrue(section["engines_satisfied"])
        self.assertEqual(section["exit_code"], 0)

    def test_mismatched_node_fails_with_actionable_reason(self):
        repo = self._catalog_repo(
            "catalog-mismatch",
            '{"name": "fixture-catalog", "private": true,'
            '"engines": {"node": ">=99.0"},'
            '"scripts": {"check": "node -e \\"process.exit(0)\\""}}')
        report = self.tmp / "catalog-mismatch-reports" / "report.json"
        proc = run_runner(repo, "--report", str(report),
                          "--skip-ctest", "--timeout", "120")
        self.assertEqual(proc.returncode, 1)
        data = json.loads(report.read_text())
        section = data["catalog"]
        self.assertEqual(section["status"], "engines-mismatch")
        reasons = " ".join(data["overall"]["reasons"])
        self.assertIn("engines", reasons)
        self.assertIn("99.0", reasons)
        # No check run was claimed for an unsupported Node.
        self.assertNotIn("exit_code", section)

    def test_missing_engines_declaration_fails(self):
        repo = self._catalog_repo(
            "catalog-noengines",
            '{"name": "fixture-catalog", "private": true,'
            '"engines": {},'
            '"scripts": {"check": "node -e \\"process.exit(0)\\""}}')
        report = self.tmp / "catalog-noengines-reports" / "report.json"
        proc = run_runner(repo, "--report", str(report),
                          "--skip-ctest", "--timeout", "120")
        self.assertEqual(proc.returncode, 1)
        data = json.loads(report.read_text())
        self.assertEqual(data["catalog"]["status"], "engines-missing")

    def test_unparseable_package_json_fails(self):
        def layout(repo: Path) -> None:
            _green_layout(repo)
            package = repo / "scripts/tools/catalog-publisher"
            package.mkdir(parents=True, exist_ok=True)
            (package / "node_modules").mkdir(exist_ok=True)
            package.joinpath("package.json").write_text("{not json")
        repo = _init_repo(self.tmp / "catalog-badjson", layout)
        report = self.tmp / "catalog-badjson-reports" / "report.json"
        proc = run_runner(repo, "--report", str(report),
                          "--skip-ctest", "--timeout", "120")
        self.assertEqual(proc.returncode, 1)
        data = json.loads(report.read_text())
        self.assertEqual(data["catalog"]["status"], "engines-invalid")


class CatalogBeforePythonTests(RunnerFixtureTestCase):
    """The catalog gate runs before the Python suites that read its output.

    Regression for the 2026-10-06 ordering defect: the catalog section ran
    after the Python suites, so on a clean checkout — no ignored ``dist/``
    left over from a previous local build — the ultralytics_yolo
    asset/manifest snapshot suites failed to find the generated
    ``scripts/tools/catalog-publisher/dist/catalog.json`` they compare against
    (only a stale ignored ``dist/`` on a dirty development checkout masked
    the dependency; the independent clean-clone full run failed exactly
    those two suites while the catalog itself passed afterwards).  The
    fixture reproduces that shape: a fake node/npm pair whose
    ``npm run check`` builds the catalog the dependent suite reads, no
    initial ``dist/``, the maintainer launched from outside the repository
    with ``PYTHONPATH``/``PYTHONHOME`` cleared, the catalog section enabled
    and only CTest skipped.
    """

    def _fake_node_tools(self, name: str, npm_body: str):
        """Fake node/npm executables on a PATH directory (fake-tool idiom).

        The fake node reports a version inside the fixture package's
        ``engines`` range; the fake npm's behavior for ``run check`` is the
        caller's body (build the catalog, fail, ...), so the fixture never
        needs a real Node installation.
        """
        directory = self.tmp / "fake-node" / name
        directory.mkdir(parents=True, exist_ok=True)
        node = directory / "node"
        node.write_text(
            '#!/bin/sh\n'
            'case "$*" in\n'
            '  --version) echo "v22.12.0"; exit 0 ;;\n'
            '  *) exit 0 ;;\n'
            'esac\n')
        npm = directory / "npm"
        npm.write_text(f"#!/bin/sh\n{npm_body}\n")
        node.chmod(0o755)
        npm.chmod(0o755)
        return directory

    def _launch_outside(self, repo: Path, bin_dir: Path, *args: str):
        """Launch the maintainer from outside the repository with the fake
        tools first on PATH and ``PYTHONPATH``/``PYTHONHOME`` cleared."""
        launch_dir = self.tmp / "catalog-outside-launch"
        launch_dir.mkdir(exist_ok=True)
        env = {key: value for key, value in os.environ.items()
               if key not in ("PYTHONPATH", "PYTHONHOME")}
        env["PATH"] = str(bin_dir) + os.pathsep + env.get("PATH", "")
        return subprocess.run(
            [sys.executable, str(RUNNER), "--repo", str(repo),
             "--python", sys.executable, *args],
            capture_output=True, text=True, timeout=300,
            cwd=str(launch_dir), env=env)

    def test_catalog_check_builds_catalog_before_dependent_suite(self):
        repo = _init_repo(self.tmp / "catalog-before-python",
                          _catalog_dependent_layout)
        # The clean-clone shape: no generated catalog exists yet.
        self.assertFalse((repo / CATALOG_MARKER_RELATIVE).exists())
        log = self.tmp / "catalog-before-python" / "npm-invocations.log"
        bin_dir = self._fake_node_tools(
            "before-python",
            'case "$*" in\n'
            '  "run check")\n'
            '    mkdir -p dist\n'
            '    printf \'{"marker": "fixture-catalog", "models": []}\\n\''
            ' > dist/catalog.json\n'
            f'    echo "run check" >> {log}\n'
            '    exit 0 ;;\n'
            '  *) exit 0 ;;\n'
            'esac')
        out = self.tmp / "catalog-before-python" / "reports"
        report = out / "report.json"
        proc = self._launch_outside(repo, bin_dir, "--report", str(report),
                                    "--skip-ctest", "--timeout", "120")
        data = self.load_report(proc, report)
        self.assertEqual(data["overall"]["status"], "passed")
        self.assertEqual(data["overall"]["reasons"], [])
        # The catalog section ran and passed — exactly once, no repetition.
        self.assertEqual(data["catalog"]["status"], "passed")
        self.assertEqual(data["catalog"]["exit_code"], 0)
        self.assertEqual(log.read_text().splitlines(), ["run check"])
        # The dependent suite executed its real test against the catalog the
        # check generated: it can only pass if the build happened first.
        suite = self.suite_by_dir(data, "samples/vision/yolo/tests")
        self.assertEqual(suite["status"], "passed")
        self.assertEqual(suite["tests"], 1)
        self.assertEqual(suite["failures"], 0)
        self.assertEqual(suite["errors"], 0)
        # True executed count: every discovered suite ran its real tests —
        # nothing was filtered away to manufacture the pass (green layout:
        # 13 discovered-and-run, including the optional-export skip, plus
        # the dependent suite; one optional-missing torch module, as
        # everywhere in these fixtures).
        self.assertEqual(data["python_suites"]["selected"],
                         data["python_suites"]["discovered"])
        self.assertEqual(data["python_suites"]["totals"],
                         {"tests": 14, "failures": 0, "errors": 0,
                          "skipped": 1, "optional_missing_modules": 1})
        # The generated catalog is ignored build output: no drift, and the
        # checkout stays clean after the run.
        self.assertFalse(data["source"]["content_drift"])
        self.assertEqual(data["source"]["dirty_files_end"], [])
        marker = repo / CATALOG_MARKER_RELATIVE
        self.assertTrue(marker.is_file())
        self.assertEqual(
            json.loads(marker.read_text())["marker"], "fixture-catalog")
        # CTest is the only reduced section: the catalog gate stayed on.
        self.assertEqual(data["options"]["skipped_sections"], ["ctest"])

    def test_failing_catalog_check_fails_the_run(self):
        # A catalog check that exits nonzero fails the whole run with an
        # explicit reason — the earlier position never softens failure
        # reporting (engines enforcement is covered above; this is the
        # command-failure path).
        repo = _init_repo(self.tmp / "catalog-fails",
                          _catalog_dependent_layout)
        bin_dir = self._fake_node_tools(
            "check-fails",
            'case "$*" in\n'
            '  "run check") echo "fixture catalog check failed" >&2; exit 7 ;;\n'
            '  *) exit 0 ;;\n'
            'esac')
        report = self.tmp / "catalog-fails" / "report.json"
        proc = self._launch_outside(repo, bin_dir, "--report", str(report),
                                    "--skip-ctest", "--timeout", "120")
        self.assertEqual(proc.returncode, 1)
        data = json.loads(report.read_text())
        self.assertEqual(data["catalog"]["status"], "failed")
        self.assertEqual(data["catalog"]["exit_code"], 7)
        reasons = " ".join(data["overall"]["reasons"])
        self.assertIn("catalog check failed (exit 7)", reasons)


class SampleInventoryTests(RunnerFixtureTestCase):
    """Discovery is validated against the accepted (required) inventory.

    The accepted inventory is required proof of coverage: a checkout that
    does not carry it — or carries an empty/malformed one — fails the gate
    with an explicit reason instead of silently disabling the coverage
    check.  Malformed rows are reported, never filtered away.
    """

    INVENTORY_SAMPLES = GREEN_INVENTORY_SAMPLES

    @staticmethod
    def _row(sample: str) -> dict:
        # Shaped like the real accepted record: the sample-relative
        # identifier plus the descriptive fields the inventory carries.
        domain, name = sample.split("/", 1)
        return {"sample": sample, "domain": domain, "batch": "fixture",
                "entry": "runtime/python/main.py"}

    def _inventory_layout(self, rows=None, document=None):
        if document is None:
            if rows is None:
                rows = [self._row(sample) for sample in self.INVENTORY_SAMPLES]
            document = {"rows": rows, "filesystem_check": {}}

        def layout(repo: Path) -> None:
            _green_layout(repo)
            _write(repo / SAMPLE_INVENTORY_RELPATH,
                   document if isinstance(document, str)
                   else json.dumps(document))
        return layout

    def test_coverage_ok_when_inventory_matches_tree(self):
        repo = _init_repo(self.tmp / "inventory-ok", self._inventory_layout())
        report = self.tmp / "inventory-ok" / "report.json"
        proc = run_runner(repo, "--report", str(report),
                          "--skip-ctest", "--skip-catalog", "--timeout", "120")
        data = self.load_report(proc, report)
        coverage = data["sample_coverage"]
        self.assertEqual(coverage["status"], "ok")
        self.assertEqual(coverage["expected_samples"], 5)
        self.assertEqual(coverage["missing_samples"], [])
        self.assertEqual(coverage["missing_tests_dirs"], [])
        self.assertEqual(coverage["extra_samples"], [])

    def test_deleted_sample_tests_directory_is_detected(self):
        # Real-shaped fixture: a repository carrying the accepted inventory
        # whose tree then loses a sample tests directory.
        repo = _init_repo(self.tmp / "inventory-missing",
                          self._inventory_layout())
        shutil.rmtree(repo / "samples/vision/alpha/tests")
        _git(repo, "add", "-A")
        _git(repo, "commit", "-q", "-m", "drop alpha tests")
        report = self.tmp / "inventory-missing" / "report.json"
        proc = run_runner(repo, "--report", str(report),
                          "--skip-ctest", "--skip-catalog", "--timeout", "120")
        self.assertEqual(proc.returncode, 1)
        data = json.loads(report.read_text())
        coverage = data["sample_coverage"]
        self.assertEqual(coverage["status"], "failed")
        self.assertIn("vision/alpha", coverage["missing_tests_dirs"])
        reasons = " ".join(data["overall"]["reasons"])
        self.assertIn("vision/alpha", reasons)
        self.assertIn("sample inventory", reasons)

    def test_extra_sample_directory_is_detected(self):
        def layout(repo: Path) -> None:
            self._inventory_layout()(repo)
            _write(
                repo / "samples/vision/zzz/tests/test_new.py",
                """
                import unittest

                class NewTests(unittest.TestCase):
                    def test_new(self):
                        self.assertTrue(True)
                """,
            )
        repo = _init_repo(self.tmp / "inventory-extra", layout)
        report = self.tmp / "inventory-extra" / "report.json"
        proc = run_runner(repo, "--report", str(report),
                          "--skip-ctest", "--skip-catalog", "--timeout", "120")
        self.assertEqual(proc.returncode, 1)
        data = json.loads(report.read_text())
        coverage = data["sample_coverage"]
        self.assertEqual(coverage["status"], "failed")
        self.assertIn("vision/zzz", coverage["extra_samples"])

    def test_broken_inventory_fails_explicitly(self):
        repo = _init_repo(self.tmp / "inventory-broken",
                          self._inventory_layout(document="{ not json"))
        report = self.tmp / "inventory-broken" / "report.json"
        proc = run_runner(repo, "--report", str(report),
                          "--skip-ctest", "--skip-catalog", "--timeout", "120")
        self.assertEqual(proc.returncode, 1)
        data = json.loads(report.read_text())
        coverage = data["sample_coverage"]
        self.assertEqual(coverage["status"], "invalid")
        self.assertIn("unparseable", coverage["note"])
        reasons = " ".join(data["overall"]["reasons"])
        self.assertIn("sample inventory", reasons)

    def test_absent_inventory_fails_the_gate(self):
        # The accepted inventory is required: deleting it must fail the run
        # with an explicit structured reason — a missing proof can never be
        # reported as success (or full CI equivalence).
        def layout(repo: Path) -> None:
            _green_layout(repo)
            (repo / SAMPLE_INVENTORY_RELPATH).unlink()
        repo = _init_repo(self.tmp / "inventory-absent", layout)
        report = self.tmp / "inventory-absent" / "report.json"
        proc = run_runner(repo, "--report", str(report),
                          "--skip-ctest", "--skip-catalog", "--timeout", "120")
        self.assertEqual(proc.returncode, 1)
        data = json.loads(report.read_text())
        coverage = data["sample_coverage"]
        self.assertEqual(coverage["status"], "not-present")
        self.assertIn(SAMPLE_INVENTORY_RELPATH, coverage["note"])
        reasons = " ".join(data["overall"]["reasons"])
        self.assertIn(SAMPLE_INVENTORY_RELPATH, reasons)
        self.assertIn("sample inventory", reasons)
        self.assertIn("requires the accepted inventory", reasons)
        self.assertIn("cannot claim full CI equivalence", reasons)

    def test_empty_rows_inventory_is_invalid(self):
        repo = _init_repo(self.tmp / "inventory-empty",
                          self._inventory_layout(rows=[]))
        report = self.tmp / "inventory-empty" / "report.json"
        proc = run_runner(repo, "--report", str(report),
                          "--skip-ctest", "--skip-catalog", "--timeout", "120")
        self.assertEqual(proc.returncode, 1)
        data = json.loads(report.read_text())
        coverage = data["sample_coverage"]
        self.assertEqual(coverage["status"], "invalid")
        self.assertIn("empty", coverage["note"].lower())
        reasons = " ".join(data["overall"]["reasons"])
        self.assertIn("sample inventory", reasons)

    def test_malformed_rows_are_reported_not_filtered(self):
        # Non-object rows, missing identifiers, invalid sample-relative ids
        # and duplicates are reported as invalid rows — the check never
        # filters malformed rows away to manufacture a pass.
        rows = [
            self._row("vision/alpha"),
            "vision/beta-not-an-object",
            {"domain": "vision"},                      # no sample key
            {"sample": ""},                            # empty identifier
            {"sample": "vision"},                      # no name component
            {"sample": "/vision/alpha"},               # absolute path
            {"sample": "vision/../escape"},            # traversal
            {"sample": "unknown-domain/beta"},         # not a sample domain
            self._row("vision/beta"),
            self._row("vision/beta"),                  # duplicate
        ]
        repo = _init_repo(self.tmp / "inventory-malformed",
                          self._inventory_layout(rows=rows))
        report = self.tmp / "inventory-malformed" / "report.json"
        proc = run_runner(repo, "--report", str(report),
                          "--skip-ctest", "--skip-catalog", "--timeout", "120")
        self.assertEqual(proc.returncode, 1)
        data = json.loads(report.read_text())
        coverage = data["sample_coverage"]
        self.assertEqual(coverage["status"], "invalid")
        invalid = coverage["invalid_rows"]
        self.assertEqual(len(invalid), 8)
        reported_rows = {entry["row"] for entry in invalid}
        # Only the two well-formed unique rows are absent from the report.
        self.assertEqual(reported_rows, set(range(10)) - {0, 8})
        reasons = " ".join(data["overall"]["reasons"])
        self.assertIn("duplicate", reasons)
        self.assertIn("sample inventory", reasons)


class PureFunctionTests(unittest.TestCase):
    """Unit coverage for classification and parsing helpers."""

    @classmethod
    def setUpClass(cls):
        cls.module = _import_runner()

    def test_classify_skip_reason_categories(self):
        classify = self.module.classify_skip_reason
        self.assertEqual(classify("C++17 compiler unavailable; entry not-run"),
                         "native_prerequisite")
        self.assertEqual(classify("nlohmann json headers unavailable; chat app not-run"),
                         "native_prerequisite")
        self.assertEqual(classify("host gflags unavailable (/opt/x); entry not-run"),
                         "native_prerequisite")
        self.assertEqual(classify("Native compiler required"),
                         "native_prerequisite")
        self.assertEqual(classify("host C++ compiler unavailable"),
                         "native_prerequisite")
        self.assertEqual(classify("C++ compiler required for native input protocol tests"),
                         "native_prerequisite")
        self.assertEqual(classify("Torch and FunASR export dependencies required"),
                         "unexpected")
        self.assertEqual(classify("ONNX conversion dependencies required"),
                         "unexpected")
        self.assertEqual(classify("Git unavailable"), "unexpected")
        self.assertEqual(
            classify("board SDK present: import-failure path not reachable"),
            "conditional")

    def test_classify_skip_scopes_framework_skips_by_identity(self):
        classify = self.module.classify_skip
        paraformer = ("samples/speech/paraformer/tests",
                      "test_export_stages.ExportStages.test_predictor")
        self.assertEqual(
            classify(paraformer[1],
                     "Torch and FunASR export dependencies required",
                     suite_dir=paraformer[0]),
            "optional_export")
        # Same skip words in an unrelated runtime/model suite: rejected.
        self.assertEqual(
            classify("test_opt.OptionalTests.test_opt",
                     "Torch and FunASR export dependencies required",
                     suite_dir="samples/vision/eps/tests"),
            "unexpected")
        # Same module name in the wrong suite directory: rejected.
        self.assertEqual(
            classify(paraformer[1],
                     "Torch and FunASR export dependencies required",
                     suite_dir="samples/vision/eps/tests"),
            "unexpected")
        # No suite scope at all: rejected.
        self.assertEqual(
            classify(paraformer[1],
                     "Torch and FunASR export dependencies required"),
            "unexpected")
        self.assertEqual(
            classify("test_runtime.RuntimeTests.test_x",
                     "ultralytics not installed for runtime regression",
                     suite_dir="samples/vision/yoloe/tests"),
            "unexpected")
        self.assertEqual(
            classify(paraformer[1], "C++ compiler unavailable",
                     suite_dir=paraformer[0]),
            "native_prerequisite")

    def test_format_error_accepts_exc_info_and_bare_exception(self):
        format_error = self.module._format_error
        try:
            raise ValueError("boom-tuple")
        except ValueError:
            tuple_form = format_error(sys.exc_info())
        self.assertIn("boom-tuple", tuple_form)
        self.assertIn("ValueError", tuple_form)
        try:
            raise KeyboardInterrupt("boom-bare")
        except KeyboardInterrupt as exc:
            bare_form = format_error(exc)
        self.assertIn("boom-bare", bare_form)
        self.assertIn("KeyboardInterrupt", bare_form)

    def test_parse_ctest_output_supports_both_formats(self):
        parse = self.module.parse_ctest_output
        self.assertEqual(
            parse("100% tests passed, 0 tests failed out of 12\n"),
            {"total": 12, "failed": 0, "passed": 12})
        # CMake >= 4 wording without the failed clause.
        self.assertEqual(
            parse("100% tests passed out of 19\n"),
            {"total": 19, "failed": 0, "passed": 19})
        self.assertEqual(
            parse("96% tests passed, 1 tests failed out of 12\n"),
            {"total": 12, "failed": 1, "passed": 11})
        self.assertIsNone(parse("no summary here"))

    def test_parse_vitest_summary(self):
        parse = self.module.parse_vitest_summary
        text = " Tests  1 failed | 135 passed (136)\n"
        self.assertEqual(parse(text),
                         {"total": 136, "passed": 135, "failed": 1})
        self.assertEqual(parse(" Tests  136 passed (136)\n"),
                         {"total": 136, "passed": 136, "failed": 0})
        self.assertIsNone(parse("no tests ran"))

    def test_missing_optional_dependency_detection(self):
        extract = self.module.missing_optional_dependency
        self.assertEqual(
            extract("ModuleNotFoundError: No module named 'torch'"), "torch")
        self.assertEqual(
            extract("ImportError: No module named 'funasr'"), "funasr")
        self.assertIsNone(extract("ModuleNotFoundError: No module named 'numpy'"))
        self.assertIsNone(extract("AssertionError: boom"))

    def test_effective_cmake_defines_track_scope_reduction(self):
        effective = self.module.effective_cmake_defines
        merged, reductions = effective({})
        self.assertEqual(merged["asr-cpp-tests"]["ASR_CLI_TESTS"], "ON")
        # The added projects' full safe defaults are part of every merge.
        self.assertEqual(
            merged["paraformer-cpp-tests"],
            {"PARAFORMER_BUILD_TESTS": "ON", "PARAFORMER_BUILD_IO": "ON",
             "PARAFORMER_SANITIZERS": "ON", "PARAFORMER_BUILD_SDK": "OFF",
             "PARAFORMER_BUILD_CLI": "OFF"})
        self.assertEqual(
            merged["himloco-cpp-tests"],
            {"HIMLOCO_BUILD_TESTS": "ON", "HIMLOCO_BUILD_SDK": "OFF",
             "HIMLOCO_BUILD_CLI": "OFF"})
        self.assertEqual(reductions, [])
        merged, reductions = effective(
            {"cmake_defines": {"yoloe-cpp-tests": {"YOLOE_TEST_OPENCV": "OFF"}}})
        self.assertEqual(merged["yoloe-cpp-tests"]["YOLOE_TEST_OPENCV"], "OFF")
        self.assertEqual(reductions,
                         ["yoloe-cpp-tests:YOLOE_TEST_OPENCV=OFF"])
        # Disabling a required host test/IO/sanitizer flag of an added
        # project is a scope reduction too.
        merged, reductions = effective(
            {"cmake_defines": {"himloco-cpp-tests":
                               {"HIMLOCO_BUILD_TESTS": "OFF"}}})
        self.assertEqual(reductions,
                         ["himloco-cpp-tests:HIMLOCO_BUILD_TESTS=OFF"])
        # Restating a mandated OFF vendor switch is a no-op, not a reduction.
        merged, reductions = effective(
            {"cmake_defines": {"paraformer-cpp-tests":
                               {"PARAFORMER_BUILD_SDK": "OFF"}}})
        self.assertEqual(reductions, [])
        self.assertEqual(merged["paraformer-cpp-tests"]["PARAFORMER_BUILD_SDK"],
                         "OFF")
        # Adding an unrelated define is not a scope reduction.
        merged, reductions = effective(
            {"cmake_defines": {"gemma4-e2b-native": {"SOME_FLAG": "ON"}}})
        self.assertEqual(reductions, [])
        self.assertEqual(merged["gemma4-e2b-native"]["SOME_FLAG"], "ON")

    def test_prohibited_cmake_overrides_name_vendor_switches(self):
        prohibited = self.module.prohibited_cmake_overrides
        self.assertEqual(prohibited({}), [])
        # Restating the mandated OFF value is fine, and so are unrelated
        # or scope-reducing defines on other projects.
        self.assertEqual(prohibited({"cmake_defines": {
            "paraformer-cpp-tests": {"PARAFORMER_BUILD_SDK": "OFF"}}}), [])
        self.assertEqual(prohibited({"cmake_defines": {
            "yoloe-cpp-tests": {"YOLOE_TEST_OPENCV": "OFF"}}}), [])
        reasons = prohibited({"cmake_defines": {
            "paraformer-cpp-tests": {"PARAFORMER_BUILD_SDK": "ON",
                                     "PARAFORMER_BUILD_CLI": "1"},
            "himloco-cpp-tests": {"HIMLOCO_BUILD_SDK": "TRUE"}}})
        self.assertEqual(len(reasons), 3, reasons)
        joined = "\n".join(reasons)
        self.assertIn("paraformer-cpp-tests:PARAFORMER_BUILD_SDK=ON", joined)
        self.assertIn("paraformer-cpp-tests:PARAFORMER_BUILD_CLI=1", joined)
        self.assertIn("himloco-cpp-tests:HIMLOCO_BUILD_SDK=TRUE", joined)
        self.assertIn("never builds the vendor SDK", joined)
        for reason in reasons:
            self.assertIn("rejected", reason)

    def test_parse_cmake_defines_reject_typed_cache_keys(self):
        # The advertised grammar is PROJECT:VAR=VALUE only.  A typed CMake
        # cache spelling (VAR:BOOL, VAR:STRING, VAR:PATH, ...) must be
        # rejected with an actionable message naming the supported form —
        # never accepted as a literal key such as "VAR:BOOL".
        parse = self.module._parse_cmake_defines
        for item in ("paraformer-cpp-tests:PARAFORMER_BUILD_SDK:BOOL=ON",
                     "himloco-cpp-tests:HIMLOCO_BUILD_CLI:BOOL=1",
                     "paraformer-cpp-tests:PARAFORMER_SANITIZERS:BOOL=OFF",
                     "gemma4-e2b-native:GEMMA_PIN:STRING=deadbeef",
                     "asr-cpp-tests:ASR_FRONTEND:PATH=/opt/local",
                     "yoloe-cpp-tests:YOLOE_TEST_OPENCV:INTERNAL=1"):
            with self.subTest(item=item):
                with self.assertRaises(SystemExit) as caught:
                    parse([item])
                message = str(caught.exception)
                self.assertIn(item, message)
                self.assertIn("PROJECT:VAR=VALUE", message)
                self.assertIn("typed", message.lower())
        # Untyped parsing is unchanged: several projects, '=' inside the
        # value, and a repeated key keeping the last value.
        self.assertEqual(
            parse(["yoloe-cpp-tests:YOLOE_TEST_OPENCV=OFF",
                   "paraformer-cpp-tests:PARAFORMER_BUILD_SDK=OFF",
                   "asr-cpp-tests:ASR_NOTE=a=b",
                   "yoloe-cpp-tests:YOLOE_TEST_OPENCV=ON"]),
            {"yoloe-cpp-tests": {"YOLOE_TEST_OPENCV": "ON"},
             "paraformer-cpp-tests": {"PARAFORMER_BUILD_SDK": "OFF"},
             "asr-cpp-tests": {"ASR_NOTE": "a=b"}})

    def test_prohibited_cmake_overrides_reject_typed_enablings(self):
        # A typed spelling of a mandated-OFF vendor/production switch must
        # not slip past the exact-key comparison: every typed enabling is a
        # rejection reason naming the typed key it arrived as.
        prohibited = self.module.prohibited_cmake_overrides
        reasons = prohibited({"cmake_defines": {
            "paraformer-cpp-tests": {"PARAFORMER_BUILD_SDK:BOOL": "ON",
                                     "PARAFORMER_BUILD_CLI:BOOL": "true"},
            "himloco-cpp-tests": {"HIMLOCO_BUILD_SDK:BOOL": "ON",
                                  "HIMLOCO_BUILD_CLI:BOOL": "1"}}})
        self.assertEqual(len(reasons), 4, reasons)
        joined = "\n".join(reasons)
        for expected in ("paraformer-cpp-tests:PARAFORMER_BUILD_SDK:BOOL=ON",
                         "paraformer-cpp-tests:PARAFORMER_BUILD_CLI:BOOL=true",
                         "himloco-cpp-tests:HIMLOCO_BUILD_SDK:BOOL=ON",
                         "himloco-cpp-tests:HIMLOCO_BUILD_CLI:BOOL=1"):
            self.assertIn(expected, joined)
        self.assertIn("never builds the vendor SDK", joined)
        self.assertIn("typed", joined.lower())
        # A typed restatement of the mandated OFF value enables nothing, so
        # this guard reports nothing — the parser rejects the typed syntax
        # itself before it can ever reach this function through the CLI.
        self.assertEqual(prohibited({"cmake_defines": {
            "paraformer-cpp-tests": {"PARAFORMER_BUILD_SDK:BOOL": "OFF"}}}),
            [])

    def test_effective_cmake_defines_classify_typed_off_as_reduction(self):
        # A typed OFF of a required default-ON test/IO/sanitizer flag must
        # never pass as CI-equivalent scope: it is recorded as a reduction
        # naming the typed spelling it was seen with, and the typed key is
        # kept as given — never rewritten into a supported bare define.
        effective = self.module.effective_cmake_defines
        merged, reductions = effective({"cmake_defines": {
            "paraformer-cpp-tests": {"PARAFORMER_BUILD_TESTS:BOOL": "OFF",
                                     "PARAFORMER_BUILD_IO:BOOL": "OFF",
                                     "PARAFORMER_SANITIZERS:BOOL": "OFF"},
            "himloco-cpp-tests": {"HIMLOCO_BUILD_TESTS:BOOL": "OFF"}}})
        self.assertEqual(
            reductions,
            ["paraformer-cpp-tests:PARAFORMER_BUILD_TESTS:BOOL=OFF",
             "paraformer-cpp-tests:PARAFORMER_BUILD_IO:BOOL=OFF",
             "paraformer-cpp-tests:PARAFORMER_SANITIZERS:BOOL=OFF",
             "himloco-cpp-tests:HIMLOCO_BUILD_TESTS:BOOL=OFF"])
        self.assertEqual(
            merged["paraformer-cpp-tests"]["PARAFORMER_SANITIZERS:BOOL"],
            "OFF")
        # A typed enabling of a mandated-OFF switch is a guard rejection,
        # not a scope reduction; a typed ON restating a default-ON flag
        # changes nothing and is neither.
        for defines in ({"PARAFORMER_BUILD_SDK:BOOL": "ON"},
                        {"PARAFORMER_SANITIZERS:BOOL": "ON"}):
            with self.subTest(defines=defines):
                _, reductions = effective(
                    {"cmake_defines": {"paraformer-cpp-tests": defines}})
                self.assertEqual(reductions, [])

    def test_prohibited_cmake_overrides_reject_padded_false_tokens(self):
        # CMake preserves the leading whitespace of a -D value (only
        # trailing whitespace is stripped at caching), so " OFF " leaves
        # the switch enabled in fact (real CMake 4.4.4, independent-cmake-
        # boolean-whitespace; trailing-whitespace probes in
        # host_cmake_false_scope).  The guard must compare the token
        # exactly as it is emitted on the -D command line: every
        # whitespace-padded false-looking value of a mandated-OFF
        # vendor/production switch is rejected fail-closed — an enabling
        # value in fact when leading-padded, and never trusted either way
        # — while only exact, unpadded false-constant spellings stay
        # accepted no-ops.
        prohibited = self.module.prohibited_cmake_overrides
        switches = (
            ("paraformer-cpp-tests", "PARAFORMER_BUILD_SDK"),
            ("paraformer-cpp-tests", "PARAFORMER_BUILD_CLI"),
            ("himloco-cpp-tests", "HIMLOCO_BUILD_SDK"),
            ("himloco-cpp-tests", "HIMLOCO_BUILD_CLI"),
        )
        padded = (" OFF ", " false ", " 0 ", " NO ",
                  "OFF ", " off", "\tFALSE\n", " 0")
        for project, key in switches:
            for value in padded:
                with self.subTest(define=f"{project}:{key}={value!r}"):
                    reasons = prohibited({"cmake_defines": {
                        project: {key: value}}})
                    self.assertEqual(len(reasons), 1, reasons)
                    self.assertIn(f"{project}:{key}={value}", reasons[0])
                    self.assertIn("rejected", reasons[0])
        # The rejection of a padded false-looking token names the mechanism
        # so the call site can be fixed to the exact token.
        reasons = prohibited({"cmake_defines": {
            "paraformer-cpp-tests": {"PARAFORMER_BUILD_SDK": " OFF "}}})
        self.assertIn("trim", reasons[0])
        # Exact case-insensitive false spellings of every supported variant
        # remain no-op restatements for every mandated-OFF switch...
        for project, key in switches:
            for value in ("OFF", "off", "Off", "FALSE", "False",
                          "0", "NO", "no", "nO"):
                with self.subTest(exact=f"{project}:{key}={value!r}"):
                    self.assertEqual(prohibited({"cmake_defines": {
                        project: {key: value}}}), [])
        # ...while enabling values are rejected with or without padding.
        for project, key in switches:
            for value in ("ON", " on ", "TRUE", " 1", "YES "):
                with self.subTest(enabling=f"{project}:{key}={value!r}"):
                    self.assertNotEqual(prohibited({"cmake_defines": {
                        project: {key: value}}}), [])

    def test_prohibited_cmake_overrides_leave_unprotected_values_alone(self):
        # Fail-closed exact-token matching applies to the mandated-OFF
        # vendor/production switches only: padded or arbitrary values of
        # other keys are not this guard's business.  A padded false token on
        # a default-ON flag stays truthy to CMake, so the flag is not
        # actually off and no scope is lost there (the reduction classifier
        # compares the same raw value — see below).
        prohibited = self.module.prohibited_cmake_overrides
        self.assertEqual(prohibited({"cmake_defines": {
            "yoloe-cpp-tests": {"YOLOE_TEST_OPENCV": " OFF "},
            "paraformer-cpp-tests": {"PARAFORMER_BUILD_TESTS": " false ",
                                     "PARAFORMER_SANITIZERS": " 0 "},
            "gemma4-e2b-native": {"SOME_FLAG": " padded custom value ",
                                  "GEMMA_PIN": "deadbeef"},
            "asr-cpp-tests": {"ASR_NOTE": "a=b"}}}), [])

    def test_prohibited_cmake_overrides_follow_cmake_false_constants(self):
        # The mandated-OFF guard classifies with real CMake's false-constant
        # semantics (verified against CMake 4.4.4 — independent-cmake-false-
        # constants plus the extended spelling probe): "", N, IGNORE and
        # the named constants (case-insensitive) and the exact NOTFOUND /
        # *-NOTFOUND spellings (case-sensitive) leave the vendor/production
        # switch OFF in fact, so restating one is an accepted no-op.  Every
        # other spelling is rejected fail-closed: the lowercase NOTFOUND
        # spellings are truthy to CMake (enabling values in fact), and no
        # padded spelling is ever trusted — CMake keeps leading whitespace,
        # and the guard does not guess which padding CMake might strip —
        # so padded tokens are rejected with a reason naming the padding.
        prohibited = self.module.prohibited_cmake_overrides
        switches = (
            ("paraformer-cpp-tests", "PARAFORMER_BUILD_SDK"),
            ("paraformer-cpp-tests", "PARAFORMER_BUILD_CLI"),
            ("himloco-cpp-tests", "HIMLOCO_BUILD_SDK"),
            ("himloco-cpp-tests", "HIMLOCO_BUILD_CLI"),
        )
        for project, key in switches:
            for value in ("", "N", "n", "IGNORE", "ignore",
                          "NOTFOUND", "missing-NOTFOUND",
                          "HIMLOCO_SDK-NOTFOUND"):
                with self.subTest(noop=f"{project}:{key}={value!r}"):
                    self.assertEqual(prohibited({"cmake_defines": {
                        project: {key: value}}}), [])
            for value in ("notfound", "NotFound", "X-notfound", "X-Notfound",
                          " N ", "\tIGNORE", "NOTFOUND ", "OFF ",
                          " missing-NOTFOUND "):
                with self.subTest(rejected=f"{project}:{key}={value!r}"):
                    reasons = prohibited({"cmake_defines": {
                        project: {key: value}}})
                    self.assertEqual(len(reasons), 1, reasons)
                    self.assertIn(f"{project}:{key}={value}", reasons[0])
                    self.assertIn("rejected", reasons[0])
        # The padded rejection names the padding mechanism so the call
        # site can be fixed to the exact token.
        reasons = prohibited({"cmake_defines": {
            "paraformer-cpp-tests": {"PARAFORMER_BUILD_SDK": " N "}}})
        self.assertIn("whitespace", reasons[0])

    def test_effective_cmake_defines_compare_values_as_emitted(self):
        # Scope-reduction classification compares the raw value, exactly as
        # it is emitted on the cmake -D command line and exactly as CMake
        # evaluates it: CMake preserves the leading whitespace of a -D
        # value (it strips only trailing whitespace when caching), so a
        # leading-padded false token on a default-ON flag keeps that flag
        # ON in fact — not a scope reduction, and never reported as one.
        # Exact false spellings are real reductions, and custom string/path
        # values pass through unmodified — never rewritten for comparison
        # or emission.
        effective = self.module.effective_cmake_defines
        merged, reductions = effective({"cmake_defines": {
            "paraformer-cpp-tests": {"PARAFORMER_BUILD_TESTS": " OFF "}}})
        self.assertEqual(reductions, [])
        self.assertEqual(merged["paraformer-cpp-tests"]["PARAFORMER_BUILD_TESTS"],
                         " OFF ")
        merged, reductions = effective({"cmake_defines": {
            "asr-cpp-tests": {"ASR_CLI_TESTS": "off"}}})
        self.assertEqual(reductions, ["asr-cpp-tests:ASR_CLI_TESTS=off"])
        self.assertEqual(merged["asr-cpp-tests"]["ASR_CLI_TESTS"], "off")
        merged, reductions = effective({"cmake_defines": {
            "gemma4-e2b-native": {"SOME_FLAG": " padded value "}}})
        self.assertEqual(reductions, [])
        self.assertEqual(merged["gemma4-e2b-native"]["SOME_FLAG"],
                         " padded value ")

    def test_effective_cmake_defines_classify_all_cmake_false_constants(self):
        # Real CMake 4.4.4 evaluates each of "" / N / IGNORE / NOTFOUND /
        # missing-NOTFOUND to OFF through option()+if() on an untyped -D
        # value (independent-cmake-false-constants evidence, extended by
        # the spelling/trailing-whitespace probe in host_cmake_false_scope/
        # execution/extra-cmake-aliases).  Each is a legitimate spelling
        # that turns a required default-ON test/IO/sanitizer flag off in
        # fact, so every one must be recorded as a scope reduction naming
        # the raw value — never merged as if the run were still
        # CI-equivalent.
        effective = self.module.effective_cmake_defines
        evidenced = ("", "N", "IGNORE", "NOTFOUND", "missing-NOTFOUND")
        targets = (("yoloe-cpp-tests", "YOLOE_TEST_OPENCV"),
                   ("asr-cpp-tests", "ASR_AUDIO_TESTS"),
                   ("asr-cpp-tests", "ASR_CLI_TESTS"),
                   ("paraformer-cpp-tests", "PARAFORMER_BUILD_TESTS"),
                   ("paraformer-cpp-tests", "PARAFORMER_BUILD_IO"),
                   ("paraformer-cpp-tests", "PARAFORMER_SANITIZERS"),
                   ("himloco-cpp-tests", "HIMLOCO_BUILD_TESTS"))
        for value in evidenced:
            for project, key in targets:
                with self.subTest(define=f"{project}:{key}={value!r}"):
                    merged, reductions = effective({"cmake_defines": {
                        project: {key: value}}})
                    self.assertEqual(reductions, [f"{project}:{key}={value}"])
                    self.assertEqual(merged[project][key], value)
        # The named constants compare case-insensitively; NOTFOUND — the
        # exact token and the -NOTFOUND suffix — case-sensitively; and
        # CMake strips trailing whitespace from a -D value when caching it
        # (cache probes: "OFF " is cached as OFF), so trailing-padded
        # spellings are real OFFs and reductions naming the raw value.
        for value in ("0", "OFF", "off", "Off", "oFf", "NO", "no", "No",
                      "FALSE", "False", "n", "nO", "Ignore", "ignore",
                      "NOTFOUND", "OpenCV-NOTFOUND", "HIMLOCO_SDK-NOTFOUND",
                      "OFF ", "N ", "NOTFOUND ", "missing-NOTFOUND "):
            with self.subTest(false_constant=repr(value)):
                _, reductions = effective({"cmake_defines": {
                    "paraformer-cpp-tests":
                        {"PARAFORMER_SANITIZERS": value}}})
                self.assertEqual(
                    reductions,
                    [f"paraformer-cpp-tests:PARAFORMER_SANITIZERS={value}"])
        # Leading whitespace survives CMake's -D caching, so leading- and
        # symmetrically-padded tokens keep the flag ON in fact and none of
        # them may be reported as a reduction.  The lowercase NOTFOUND
        # spellings are truthy (case-sensitive match) and behave like
        # arbitrary strings: no reduction, value unchanged.
        for value in (" OFF ", " N ", "\tno", " ignore ", " NOTFOUND",
                      " off", "notfound", "NotFound", "lib-notfound",
                      "X-notfound", "X-Notfound", "deadbeef-NOTFOUND-ish"):
            with self.subTest(truthy=repr(value)):
                merged, reductions = effective({"cmake_defines": {
                    "yoloe-cpp-tests": {"YOLOE_TEST_OPENCV": value}}})
                self.assertEqual(reductions, [])
                self.assertEqual(merged["yoloe-cpp-tests"]["YOLOE_TEST_OPENCV"],
                                 value)

    def test_version_satisfies_node_ranges(self):
        satisfies = self.module._version_satisfies
        self.assertTrue(satisfies("22.23.2", ">=22.12 <23"))
        self.assertTrue(satisfies("22.12.0", ">=22.12 <23"))
        self.assertFalse(satisfies("26.9.0", ">=22.12 <23"))
        self.assertFalse(satisfies("22.11.9", ">=22.12 <23"))
        self.assertTrue(satisfies("22.12.0", "22.12"))

    def test_requirements_scipy_markers_match_tested_environments(self):
        text = REQUIREMENTS.read_text()
        lowered = text.lower()
        lines = [line for line in text.splitlines()
                 if line.strip().startswith("scipy")]
        self.assertEqual(len(lines), 2, lines)
        darwin_line = next(line for line in lines
                           if "darwin" in line and "1.17.1" in line)
        self.assertIn('python_version >= "3.12"', darwin_line)
        general_line = next(line for line in lines
                            if line is not darwin_line)
        self.assertIn("1.10", general_line)
        # The truthfulness record: the darwin-wheel failure and the tested
        # good version are documented with the upstream reference.
        self.assertIn("1.15.3", text)
        self.assertIn("scipy/scipy/issues/25635", text)
        for package in ("ftfy", "regex", "pillow", "onnx", "onnxruntime",
                        "pycocotools"):
            self.assertIn(package, lowered)


class RealRepositoryTests(unittest.TestCase):
    """Discovery and the CTest registry checked against the actual tree."""

    @classmethod
    def setUpClass(cls):
        cls.module = _import_runner()

    def test_discovery_covers_nested_and_declared_directories(self):
        result = self.module.discover_python_suites(REPO)
        dirs = {s["dir"] for s in result["python_suites"]}
        # Nested YOLOE suites from the plan are first-class suites.
        self.assertIn("samples/vision/yoloe/conversion/tests", dirs)
        self.assertIn("samples/vision/yoloe/evaluator/tests", dirs)
        # All 51 native samples with Python suites are discovered.
        self.assertIn("samples/llm/gemma4-e2b/tests", dirs)
        self.assertIn("samples/llm/minicpm5-2b/tests", dirs)
        self.assertIn("samples/vision/ultralytics_yolo/tests", dirs)
        # Declared extras: shared tools, skills and the runner's own tests.
        self.assertIn("scripts/tools/board_validation/tests", dirs)
        self.assertIn("scripts/tools/sample_contract/tests", dirs)
        self.assertIn("skills/tests", dirs)
        self.assertIn("scripts/tools/host_validation", dirs)
        # The VLA parent-repo guard now runs with the shared suite: no
        # exclusions remain (upstream gitlinks are never initialized).
        shared = next(s for s in result["python_suites"]
                      if s["dir"] == "utils/py_utils/tests")
        self.assertNotIn("excluded_files", shared)
        # Native-only directories are recorded, not treated as Python suites.
        self.assertIn("samples/vision/yoloe/runtime/cpp/tests",
                      result["native_only_dirs"])
        self.assertIn("samples/speech/asr/runtime/cpp/tests",
                      result["native_only_dirs"])
        # Nothing unexplained: every tests directory is accounted for.
        self.assertEqual(result["anomalies"], [])
        self.assertEqual(result["missing_directories"], [])

    def test_ctest_registry_matches_real_projects(self):
        registry = self.module.CTEST_PROJECTS
        names = {project["name"] for project in registry}
        self.assertEqual(
            names,
            {"gemma4-e2b-native", "yoloe-cpp-tests", "asr-cpp-tests",
             "ultralytics-yolo-cpp-common", "paraformer-cpp-tests",
             "himloco-cpp-tests"})
        for project in registry:
            source = REPO / project["source"]
            self.assertTrue(
                (source / "CMakeLists.txt").is_file(),
                f"registered CTest project missing CMakeLists: {project['source']}")
        # The two runtime/cpp roots are registered at their actual project
        # locations, not at some derived tests subdirectory.
        sources = {project["name"]: project["source"] for project in registry}
        self.assertEqual(sources["paraformer-cpp-tests"],
                         "samples/speech/paraformer/runtime/cpp")
        self.assertEqual(sources["himloco-cpp-tests"],
                         "samples/robotics/himloco/runtime/cpp")

    def test_real_platforms_pin_is_declared_for_verification(self):
        sources = {entry["pin"] for entry in self.module.declared_pins(REPO)}
        self.assertIn(REAL_PIN, sources)

    def test_sample_coverage_validates_all_51_inventory_rows(self):
        discovery = self.module.discover_python_suites(REPO)
        coverage = self.module.validate_sample_coverage(REPO, discovery)
        self.assertEqual(coverage["status"], "ok", coverage)
        self.assertEqual(coverage["expected_samples"], 51)
        self.assertEqual(coverage["missing_samples"], [])
        self.assertEqual(coverage["missing_tests_dirs"], [])
        self.assertEqual(coverage["extra_samples"], [])

    def test_source_manifest_is_stable_and_excludes_gitlinks(self):
        first = self.module.source_manifest(REPO)
        second = self.module.source_manifest(REPO)
        self.assertEqual(first["digest"], second["digest"])
        self.assertIn("samples/vla/act", first["gitlinks"])
        self.assertNotIn("samples/vla/act", first["files_hashed_sample"])


def _import_runner():
    if not RUNNER.is_file():
        raise unittest.SkipTest("run.py not implemented yet")
    import importlib.util
    spec = importlib.util.spec_from_file_location("host_validation_run", RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    unittest.main()
