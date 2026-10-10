#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Maintainer host-validation runner for the unified RDK Model Zoo source.

Runs the complete supported host gate in one deterministic command::

    python utils/tools/host_validation/run.py --repo PATH --report PATH [--python PYTHON]

This is a maintainer test orchestrator, not a user inference CLI.  It never
downloads models, never initializes VLA submodules, never touches board SDKs
and never runs on a board.  It executes, in isolated subprocesses:

* every first-party Python ``unittest`` directory under ``samples/`` (all
  51 samples listed in the accepted coverage inventory, including nested
  suites such as
  ``samples/vision/yoloe/conversion/tests`` and ``evaluator/tests``), the
  shared suites (including ``test_vla_integration.py``, a parent-repo
  gitlink/pin integrity check that never executes upstream ACT/Pi0 code),
  ``utils/tools/board_validation``, ``utils/tools/sample_contract``, ``skills`` and this
  runner's own tests;
* the static sample-contract checker (``--scope migration --parser-mode import``);
* the existing host-safe native CTest projects with their full-gate flags on
  by default (``YOLOE_TEST_OPENCV=ON``, ``ASR_AUDIO_TESTS=ON``,
  ``ASR_CLI_TESTS=ON``, ``PARAFORMER_BUILD_TESTS=ON`` +
  ``PARAFORMER_BUILD_IO=ON`` + ``PARAFORMER_SANITIZERS=ON``,
  ``HIMLOCO_BUILD_TESTS=ON``: real OpenCV image/mask/pipeline and
  SDK-double/CLI tests, real libsndfile/libsamplerate frontends and host
  CLI tests, the sanitised Paraformer contract/pipeline/SDK-double/
  preflight/prepared-feature/host-CLI-fixture suite over Git-tracked
  evidence, and the SDK-free HIMLoco numerical policy test — no vendor SDK
  adapter, production CLI or model file is ever compiled: the SDK/CLI
  build switches are mandated OFF and cannot be enabled through
  ``--cmake-define``, which accepts only untyped ``PROJECT:VAR=VALUE``
  keys — typed CMake cache spellings (``VAR:BOOL``/``:STRING``/``:PATH``/
  ...) and whitespace-padded false tokens (``" OFF "`` — CMake keeps
  leading whitespace on ``-D`` values, so a padded token cannot be
  trusted as OFF) are rejected before any configure/build — while
  turning a default-ON flag off via any CMake false constant
  (``0``/``OFF``/``NO``/``FALSE``/``N``/``IGNORE`` and the empty value
  case-insensitively, the case-sensitive ``NOTFOUND`` and ``*-NOTFOUND``
  spellings, or trailing whitespace CMake strips itself) is recorded as
  a scope reduction naming the raw value, so the run can never stay
  CI-equivalent);
* the model-catalog check (``npm run check`` in ``utils/tools/catalog-publisher``)
  under the Node ``engines`` range the package itself declares.  This
  section runs **first**, before the Python suites: its build stage
  generates ``utils/tools/catalog-publisher/dist/catalog.json``, which the
  ultralytics_yolo asset/manifest snapshot suites compare against, so the
  catalog must exist before those suites execute (``npm ci`` remains a
  maintainer prerequisite — the runner never installs; with
  ``--skip-catalog`` no catalog is built, and those suites then need one
  generated beforehand or they fail).

Every suite runs in its own process, so identical test module names in
different samples cannot mask each other, and results are collected from a
machine-readable ``unittest`` result: exact counts, per-skip identities and
reasons, per-failure messages.  A suite with zero tests, a missing declared
directory, an unexplained directory, a crashed worker, a timeout, an
undeclared skip, a rejected native skip, a missing pinned historical commit,
an absent, malformed or mismatched accepted sample inventory or source drift
during the run can never report success.

Skip policy (strict by default, matching CI):

* ``optional_export`` — the export scope the delivery plan declares optional
  on the host, exactly: module-level Torch/FunASR/Ultralytics import
  failures in ``*/conversion/tests`` directories, and the Paraformer
  export-stage suite's conditional framework skip
  (``samples.speech.paraformer.tests.test_export_stages``).  Allowed, always
  recorded with identity and reason, never counted as executed tests.  The
  same framework names in unrelated runtime/model/native suites are
  ``unexpected`` — the optional scope never hides a dependency regression.
* ``native_prerequisite`` — missing C++ compiler, nlohmann-json, gflags,
  iconv or CMake.  Rejected in strict mode; ``--allow-native-skips`` records
  them instead (the report then says ``ci_equivalent: false``).
* ``conditional`` — legitimately conditional skips (e.g. a board SDK being
  installed makes an import-failure path unreachable).  Allowed, recorded.
* anything else is ``unexpected`` and always fails the run.

Source identity is content-based, not HEAD-based: a deterministic sha256
digest is taken over every tracked file (except mode-160000 gitlinks, whose
upstream code is not present) plus every untracked file Git's ignore rules
do not exclude, before and after the run.  Any content, dirty-state or HEAD
change during the run fails the gate, and a checkout that is not a Git
repository can never pass.  Reports and build artifacts must live outside
the source tree (``--report`` is rejected inside it) so the gate can never
pass by writing its own report into the tree.

CTest case counts are reported in their own section and are deliberately NOT
added to the Python unittest totals; a per-stage timeout, a missing
executable or corrupt discovery output fails that project (with its logs
kept) while the remaining projects still run and the report is still
written.  Exit code 0 means every section passed.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import os
import platform as host_platform
import re
import shutil
import subprocess
import sys
import time
import unittest
from datetime import datetime, timezone
from pathlib import Path

REPORT_SCHEMA = "host-validation-report/1"
RUNNER_PATH = Path(__file__).resolve()

WORKER_SENTINEL = "@@HOST_VALIDATION_WORKER_JSON@@"

#: Export-scope frameworks the delivery plan declares optional on the host.
OPTIONAL_EXPORT_DEPS = ("torch", "funasr", "ultralytics")

_OPTIONAL_EXPORT_RE = re.compile(r"(?i)\b(torch|funasr|ultralytics)\b")
_NATIVE_SKIP_RE = re.compile(
    r"(?i)(c\+\+|clang\b|\bg\+\+\b|compiler|nlohmann|gflags|iconv|cmake|"
    r"json\.hpp|native)")
#: Skips that are legitimately conditional in a healthy environment.
_CONDITIONAL_REASONS = (
    "board SDK present: import-failure path not reachable",
)

#: The single runtime suite whose conditional framework skip belongs to the
#: declared optional export scope (Torch/FunASR export-stage contracts).
#: Workers discover modules relative to their tests directory, so the skip
#: identity is the module-relative test id, scoped by the suite directory.
#: Anything else naming those frameworks is an unrelated runtime skip and
#: must fail, not silently pass.
PARAFORMER_EXPORT_SUITE_DIR = "samples/speech/paraformer/tests"
PARAFORMER_EXPORT_MODULE_PREFIX = "test_export_stages."

#: The accepted 51-row native sample inventory (Codex-reviewed readable-
#: runtime coverage record).  It is required proof of sample coverage: the
#: maintained gate validates discovery against it, so a checkout that does
#: not carry it — or carries an empty/malformed one — fails with an
#: explicit reason and can never claim full CI equivalence.
SAMPLE_INVENTORY_RELPATH = (
    "utils/tools/host_validation/sample-inventory.json")
#: First-party sample domains (``vla`` holds excluded pinned gitlinks).
SAMPLE_DOMAINS = ("vision", "llm", "robotics", "speech")

#: A valid inventory row identifies one in-repo sample by its
#: sample-relative ``domain/name`` id — no absolute paths, no traversal,
#: no nesting below the sample level.
_SAMPLE_ID_RE = re.compile(
    r"(?:vision|llm|robotics|speech)/[A-Za-z0-9][A-Za-z0-9._-]*")

#: Tool and skills suites outside ``samples/``.
EXTRA_SUITE_DIRS = (
    "utils/py_utils/tests",
    "utils/tools/board_validation/tests",
    "utils/tools/sample_contract/tests",
    "skills/tests",
)
HOST_VALIDATION_DIR = "utils/tools/host_validation"

#: The actual host-safe native CTest projects present in the tree.  Each one
#: builds against SDK doubles or plain host libraries only: no vendor SDK, no
#: model files, no downloads, no production SDK project.  Enumeration is
#: explicit so a moved or deleted project fails loudly instead of silently
#: dropping coverage.  The Paraformer and HIMLoco projects live under their
#: samples' ``runtime/cpp`` directories — a configured source directory there
#: does not imply vendor execution: their explicit default flags keep the
#: vendor SDK adapter and the production CLI OFF (see
#: :data:`MANDATORY_SAFE_DEFINES`).
CTEST_PROJECTS = (
    {
        "name": "gemma4-e2b-native",
        "source": "samples/llm/gemma4-e2b/tests/native",
        "requires_opencv": True,
        "note": "C++17 + OpenCV + full Git history (pinned platforms commit)",
    },
    {
        "name": "yoloe-cpp-tests",
        "source": "samples/vision/yoloe/runtime/cpp/tests",
        "requires_opencv": True,
        "note": "C++17, sanitizers; YOLOE_TEST_OPENCV=ON adds the real "
                "OpenCV image/mask/pipeline tests plus explicit SDK-double "
                "and CLI fixtures (on by default: the full gate)",
    },
    {
        "name": "asr-cpp-tests",
        "source": "samples/speech/asr/runtime/cpp/tests",
        "note": "C++17 contract/SDK-double/preflight tests; "
                "ASR_AUDIO_TESTS=ON and ASR_CLI_TESTS=ON add the real "
                "libsndfile/libsamplerate frontend and host CLI tests "
                "(on by default: the full gate)",
    },
    {
        "name": "ultralytics-yolo-cpp-common",
        "source": "samples/vision/ultralytics_yolo/runtime/cpp/test",
        "note": "declared host fixture: shared helpers and descriptor adapters "
                "against narrow API doubles",
    },
    {
        "name": "paraformer-cpp-tests",
        "source": "samples/speech/paraformer/runtime/cpp",
        "note": "C++17 sanitised (ASan+UBSan) host contract/pipeline tests, "
                "the SDK-double test against explicit fake headers, the "
                "preflight check over Git-tracked published-token evidence, "
                "the prepared-feature NPY/manifest probe over Git-tracked "
                "evidence (relative paths, no download/export) and the "
                "PARAFORMER_HOST_FIXTURE CLI help test — the vendor SDK "
                "adapter and production CLI stay OFF",
    },
    {
        "name": "himloco-cpp-tests",
        "source": "samples/robotics/himloco/runtime/cpp",
        "note": "SDK-free C++17 numerical policy test over the checked-in "
                "obs_history fixture (no real robot, no vendor SDK, no CLI)",
    },
)

#: CMake defines that make up the full gate.  A ``--cmake-define`` that
#: turns any of the default-ON flags OFF reduces scope: the report then
#: records the reduction and ``ci_equivalent`` becomes false.  The OFF
#: entries of the added projects are not scope choices but the mandatory
#: safety defaults themselves (see :data:`MANDATORY_SAFE_DEFINES`).
DEFAULT_CTEST_DEFINES = {
    "yoloe-cpp-tests": {"YOLOE_TEST_OPENCV": "ON"},
    "asr-cpp-tests": {"ASR_AUDIO_TESTS": "ON", "ASR_CLI_TESTS": "ON"},
    "paraformer-cpp-tests": {
        "PARAFORMER_BUILD_TESTS": "ON",
        "PARAFORMER_BUILD_IO": "ON",
        "PARAFORMER_SANITIZERS": "ON",
        "PARAFORMER_BUILD_SDK": "OFF",
        "PARAFORMER_BUILD_CLI": "OFF",
    },
    "himloco-cpp-tests": {
        "HIMLOCO_BUILD_TESTS": "ON",
        "HIMLOCO_BUILD_SDK": "OFF",
        "HIMLOCO_BUILD_CLI": "OFF",
    },
}

#: Vendor/production build switches the maintainer host gate mandates OFF.
#: These OFF defaults are mandatory safety, not a reduction of host test
#: scope: the runner never constructs a vendor SDK or production CLI build,
#: so a ``--cmake-define`` that enables one is rejected outright (before
#: any project is configured) instead of being merged or recorded as a
#: scope change.  The guard matches bare key names, typed CMake cache
#: spellings (``VAR:BOOL`` ...) are rejected at parse time, and only an
#: exact, unpadded CMake false-constant spelling is an accepted no-op
#: restatement: CMake keeps leading whitespace on ``-D`` values and the
#: guard never guesses which padding CMake might strip, so every padded
#: false-looking token is rejected fail-closed — no spelling of these
#: switches can enable a vendor build.
MANDATORY_SAFE_DEFINES = {
    "paraformer-cpp-tests": {
        "PARAFORMER_BUILD_SDK": "OFF",
        "PARAFORMER_BUILD_CLI": "OFF",
    },
    "himloco-cpp-tests": {
        "HIMLOCO_BUILD_SDK": "OFF",
        "HIMLOCO_BUILD_CLI": "OFF",
    },
}

#: Standard OpenCV CMake config locations probed when a plain
#: ``find_package(OpenCV)`` cannot resolve (Homebrew installs the config as
#: ``lib/cmake/opencv5``, not ``opencv4``).  No personal paths: only
#: documented system package roots, plus the explicit ``OPENCV_DIR`` /
#: ``OpenCV_DIR`` environment override.
OPENCV_CONFIG_CANDIDATES = (
    "/opt/homebrew/opt/opencv/lib/cmake/opencv5",
    "/opt/homebrew/opt/opencv/lib/cmake/opencv4",
    "/opt/homebrew/opt/opencv@5/lib/cmake/opencv5",
    "/opt/homebrew/opt/opencv@4/lib/cmake/opencv4",
    "/usr/local/opt/opencv/lib/cmake/opencv5",
    "/usr/local/opt/opencv/lib/cmake/opencv4",
    "/usr/local/opt/opencv@4/lib/cmake/opencv4",
    "/usr/lib/x86_64-linux-gnu/cmake/opencv4",
    "/usr/lib/aarch64-linux-gnu/cmake/opencv4",
)

#: Per-stage CTest subprocess timeouts (seconds).
CTEST_STAGE_TIMEOUTS = {"configure": 900, "build": 3600,
                        "discover": 300, "test": 3600}

#: Historical commits the tree itself declares as required Git objects.  A
#: clone that lacks one gets an explicit failure naming the fetch command —
#: never a silent skip of the suites that read pinned sources.
PIN_SOURCES = (
    ("utils/py_utils/legacy_platforms.py",
     re.compile(r'^PIN\s*=\s*"([0-9a-f]{40})"', re.M)),
    ("samples/llm/gemma4-e2b/tests/native/CMakeLists.txt",
     re.compile(r'set\s*\(\s*GEMMA_PLATFORMS_PIN\s+"([0-9a-f]{40})"')),
    ("utils/tools/catalog-publisher/sources.json", None),  # JSON walk, see below
)

_MISSING_MODULE_RE = re.compile(r"No module named '([A-Za-z0-9_.]+)'")


def classify_skip_reason(reason: str) -> str:
    """Categorize a unittest skip reason (see the module docstring).

    Framework names alone never produce ``optional_export`` here: the
    optional export scope is identified by test identity (see
    :func:`classify_skip`), so an unrelated runtime skip that happens to
    mention torch/funasr/ultralytics stays ``unexpected``.
    """
    if reason in _CONDITIONAL_REASONS:
        return "conditional"
    if _NATIVE_SKIP_RE.search(reason):
        return "native_prerequisite"
    return "unexpected"


def classify_skip(test_id: str, reason: str, suite_dir: str = None) -> str:
    """Categorize a skip with its test identity.

    ``optional_export`` applies only to the declared export scope: the
    Paraformer export-stage suite's conditional framework skip (identified
    by both the suite directory and the module-relative test id).  Loader-
    level optional misses (conversion test directories) are handled
    separately as ``optional_missing`` records.
    """
    category = classify_skip_reason(reason)
    if category != "unexpected":
        return category
    if (_OPTIONAL_EXPORT_RE.search(reason)
            and suite_dir == PARAFORMER_EXPORT_SUITE_DIR
            and test_id.startswith(PARAFORMER_EXPORT_MODULE_PREFIX)):
        return "optional_export"
    return "unexpected"


def missing_optional_dependency(error_text: str):
    """Return the optional export dependency named by an import error, if any."""
    match = _MISSING_MODULE_RE.search(error_text or "")
    if match is None:
        return None
    root = match.group(1).split(".", 1)[0]
    return root if root in OPTIONAL_EXPORT_DEPS else None


def parse_ctest_output(text: str):
    """Extract ``{total, failed, passed}`` from a CTest summary line."""
    match = re.search(
        r"([\d.]+)% tests passed(?:,\s*(\d+) tests? failed)? out of (\d+)",
        text or "",
    )
    if match is None:
        return None
    total = int(match.group(3))
    failed = int(match.group(2) or 0)
    return {"total": total, "failed": failed, "passed": total - failed}


def parse_vitest_summary(text: str):
    """Extract ``{total, passed, failed}`` from a Vitest summary line."""
    match = re.search(
        r"Tests\s+(?:(\d+) failed\s*\|\s*)?(\d+) passed(?:\s*\|\s*(\d+) skipped)?\s*\((\d+)\)",
        text or "",
    )
    if match is None:
        return None
    return {
        "failed": int(match.group(1) or 0),
        "passed": int(match.group(2)),
        "total": int(match.group(4)),
    }


def _rel(repo: Path, path: Path) -> str:
    return path.resolve().relative_to(repo.resolve()).as_posix()


def _is_export_scope(path: Path) -> bool:
    parts = path.resolve().parts
    return "conversion" in parts


def declared_pins(repo: Path):
    """Collect the pinned 40-hex commits declared by the tree itself."""
    pins = []
    seen = set()
    for relative, pattern in PIN_SOURCES:
        source = repo / relative
        if not source.is_file():
            continue
        try:
            if pattern is None:
                document = json.loads(source.read_text())
                values = _commit_pins_from_json(document)
            else:
                values = pattern.findall(source.read_text())
        except (OSError, ValueError):
            continue
        for value in values:
            value = value.lower()
            if value not in seen:
                seen.add(value)
                pins.append({"pin": value, "source": relative})
    return pins


def _commit_pins_from_json(node):
    """Yield ``mode: commit`` link_ref values (40-hex) from parsed JSON."""
    if isinstance(node, dict):
        if node.get("mode") == "commit":
            ref = str(node.get("link_ref", ""))
            if re.fullmatch(r"[0-9a-f]{40}", ref):
                yield ref
        for value in node.values():
            yield from _commit_pins_from_json(value)
    elif isinstance(node, list):
        for value in node:
            yield from _commit_pins_from_json(value)


def verify_pins(repo: Path):
    """Check each declared pin exists as a commit object in the repository."""
    results = []
    for entry in declared_pins(repo):
        done = subprocess.run(
            ["git", "-C", str(repo), "cat-file", "-e",
             f"{entry['pin']}^{{commit}}"],
            capture_output=True,
        )
        results.append({**entry, "present": done.returncode == 0})
    return results


def git_snapshot(repo: Path):
    """Capture HEAD, branch and the dirty-file list of the repository."""
    def git(*args):
        return subprocess.run(
            ["git", "-C", str(repo), *args], capture_output=True, text=True)

    commit = git("rev-parse", "HEAD")
    branch = git("rev-parse", "--abbrev-ref", "HEAD")
    status = git("status", "--porcelain")
    dirty = [line for line in (status.stdout or "").splitlines() if line]
    return {
        "commit": commit.stdout.strip() if commit.returncode == 0 else None,
        "branch": branch.stdout.strip() if branch.returncode == 0 else None,
        "dirty_files": dirty,
        "dirty_count": len(dirty),
    }


#: Individual files above this size are hashed by size marker instead of
#: content — a pathological-memory guard only; no repository file is near it.
_MANIFEST_MAX_FILE_BYTES = 64 * 1024 * 1024


def _hash_file(path: Path) -> str:
    """Deterministic content hash of one manifest file."""
    if path.stat().st_size > _MANIFEST_MAX_FILE_BYTES:
        return f"too-large:{path.stat().st_size}"
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def source_manifest(repo: Path):
    """Content identity of every first-party file in the checkout.

    Hashes all tracked files (skipping mode-160000 gitlinks: the pinned VLA
    integrations carry no upstream code in this tree) plus every untracked
    file that Git's ignore rules do not exclude (``--exclude-standard``), so
    ordinary generated/cache artifacts never count as drift.  Returns
    ``None`` when the checkout is not a usable Git repository.
    """
    def git_z(*args):
        return subprocess.run(
            ["git", "-C", str(repo), *args], capture_output=True)

    staged = git_z("ls-files", "-z", "-s")
    untracked = git_z("ls-files", "-z", "--others", "--exclude-standard")
    head = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        capture_output=True, text=True)
    if staged.returncode != 0 or untracked.returncode != 0:
        return None

    entries = []          # (path, mode, content-hash)
    gitlinks = []
    for record in staged.stdout.decode("utf-8", "surrogateescape").split("\0"):
        if not record:
            continue
        metadata, separator, path = record.partition("\t")
        if not separator:
            continue
        mode = metadata.split(" ", 1)[0]
        if mode == "160000":
            # Gitlink (submodule pin): no local content to hash.  Its
            # identity is verified by the VLA gitlink guard and pin checks.
            gitlinks.append(path)
            continue
        file = repo / path
        if not file.is_file():
            # Deleted in the worktree; still tracked.  Recorded by path so a
            # restore during the run is still a content change.
            entries.append((path, mode, "deleted"))
            continue
        try:
            entries.append((path, mode, _hash_file(file)))
        except OSError as exc:
            entries.append((path, mode, f"unreadable:{exc.errno}"))
    for path in untracked.stdout.decode("utf-8", "surrogateescape").split("\0"):
        if not path:
            continue
        file = repo / path
        if not file.is_file():
            continue
        try:
            entries.append((path, "untracked", _hash_file(file)))
        except OSError as exc:
            entries.append((path, "untracked", f"unreadable:{exc.errno}"))

    entries.sort(key=lambda item: item[0])
    digest = hashlib.sha256()
    for path, mode, content in entries:
        digest.update(mode.encode())
        digest.update(b"\0")
        digest.update(path.encode("utf-8", "surrogateescape"))
        digest.update(b"\0")
        digest.update(content.encode())
        digest.update(b"\n")
    return {
        "algorithm": "sha256 over sorted (mode, path, content-hash) of "
                     "tracked non-gitlink files plus untracked non-ignored "
                     "files",
        "digest": digest.hexdigest(),
        "file_count": len(entries),
        "gitlinks": sorted(gitlinks),
        "files_hashed_sample": [path for path, _, _ in entries[:40]],
    }


def discover_python_suites(repo: Path):
    """Enumerate every first-party Python unittest directory.

    Returns a dict with ``python_suites``, ``native_only_dirs`` (test
    directories holding only C++ sources — their coverage runs through the
    sample's Python suites or the CTest registry), ``anomalies``
    (unexplained directories) and ``missing_directories`` (declared suites
    absent from the checkout).
    """
    suites = []
    native_only = []
    anomalies = []
    missing = []
    sample_root = repo / "samples"
    vla_root = (repo / "samples" / "vla").resolve()

    candidates = []
    if sample_root.is_dir():
        for directory in sorted(sample_root.rglob("tests")):
            if not directory.is_dir():
                continue
            if vla_root in directory.resolve().parents or \
                    directory.resolve() == vla_root:
                continue  # pinned VLA gitlink integrations stay uninitialized
            candidates.append(directory)

    for directory in candidates + [repo / rel for rel in EXTRA_SUITE_DIRS]:
        if not directory.is_dir():
            continue
        rel = _rel(repo, directory)
        test_files = sorted(
            p.name for p in directory.glob("test*.py") if p.is_file())
        has_native_markers = (
            any(directory.glob("*.cc")) or any(directory.glob("*.cpp"))
            or (directory / "CMakeLists.txt").is_file())
        if test_files:
            # Every discovered file runs, including the shared suite's
            # test_vla_integration.py: a parent-repo integrity check that
            # never executes upstream ACT/Pi0 code.
            suites.append({"dir": rel})
        elif has_native_markers:
            native_only.append(rel)
        else:
            anomalies.append(
                f"{rel}: no Python tests and no native markers")

    for rel in EXTRA_SUITE_DIRS:
        if not (repo / rel).is_dir():
            missing.append(rel)
    if not (repo / HOST_VALIDATION_DIR).is_dir():
        missing.append(HOST_VALIDATION_DIR)
    else:
        suites.append({"dir": HOST_VALIDATION_DIR})

    suites.sort(key=lambda item: item["dir"])
    return {
        "python_suites": suites,
        "native_only_dirs": sorted(native_only),
        "anomalies": anomalies,
        "missing_directories": missing,
    }


def _validate_inventory_rows(rows):
    """Validate accepted-inventory rows; malformed rows are never filtered.

    Returns ``(expected, invalid_rows)`` where ``expected`` lists the valid,
    unique sample identifiers in file order and ``invalid_rows`` carries one
    entry per malformed row (non-object row, missing/empty identifier,
    invalid sample-relative id, duplicate) with its row index and reason —
    a malformed inventory fails the check instead of being quietly reduced
    to its well-formed subset.
    """
    expected = []
    invalid = []
    first_seen = {}
    for index, row in enumerate(rows):
        sample = row.get("sample") if isinstance(row, dict) else None
        if not isinstance(row, dict):
            invalid.append({"row": index,
                            "reason": "row is not a JSON object",
                            "sample": None})
            continue
        if not isinstance(sample, str) or not sample.strip():
            invalid.append({"row": index,
                            "reason": "row has no non-empty sample identifier",
                            "sample": None})
            continue
        if not _SAMPLE_ID_RE.fullmatch(sample):
            invalid.append({
                "row": index,
                "reason": "sample identifier is not a valid sample-relative "
                          "domain/name id",
                "sample": sample})
            continue
        if sample in first_seen:
            invalid.append({
                "row": index,
                "reason": f"duplicate sample identifier "
                          f"(first declared at row {first_seen[sample]})",
                "sample": sample})
            continue
        first_seen[sample] = index
        expected.append(sample)
    return expected, invalid


def validate_sample_coverage(repo: Path, discovery):
    """Validate discovery against the accepted native sample inventory.

    The inventory (``docs/releases/unified-migration/
    2026-10-05-all-sample-coverage.json``) is the Codex-reviewed 51-row
    record of in-repo samples and is **required**: every expected sample
    directory must exist and carry a discovered ``tests`` directory, and no
    extra first-party sample may appear beside them — so a sample or
    ``tests`` directory the inventory lists can never be deleted without
    the gate failing.  The enforced scope follows the reviewed inventory
    itself (no sample count is hardcoded here): an intentional future
    scope change updates that record, and this validation enforces
    whatever it lists.  A checkout
    without the inventory, or with an empty/unparseable/malformed one,
    fails with an explicit structured reason (missing proof is never
    success).  Every non-ok status carries a ``reasons`` list the report
    aggregates.
    """
    required_reason = (
        f"accepted sample inventory not present: {SAMPLE_INVENTORY_RELPATH}: "
        f"the maintained gate requires the accepted inventory record; a "
        f"checkout without it fails and cannot claim full CI equivalence")
    inventory = repo / SAMPLE_INVENTORY_RELPATH
    if not inventory.is_file():
        return {
            "status": "not-present",
            "note": f"{SAMPLE_INVENTORY_RELPATH} not present in this "
                    f"checkout; the maintained gate requires the accepted "
                    f"inventory record",
            "reasons": [required_reason],
        }
    try:
        document = json.loads(inventory.read_text())
    except ValueError as exc:
        return {"status": "invalid",
                "note": f"sample inventory unparseable: {exc}",
                "reasons": [
                    f"accepted sample inventory invalid: "
                    f"{SAMPLE_INVENTORY_RELPATH}: unparseable JSON: {exc}"]}
    rows = document.get("rows") if isinstance(document, dict) else None
    if not isinstance(rows, list):
        return {"status": "invalid",
                "note": "sample inventory has no rows array",
                "reasons": [
                    f"accepted sample inventory invalid: "
                    f"{SAMPLE_INVENTORY_RELPATH}: no rows array"]}
    if not rows:
        return {"status": "invalid",
                "note": "sample inventory rows array is empty",
                "reasons": [
                    f"accepted sample inventory invalid: "
                    f"{SAMPLE_INVENTORY_RELPATH}: rows array is empty"]}

    expected, invalid_rows = _validate_inventory_rows(rows)
    if invalid_rows:
        reasons = [
            f"accepted sample inventory invalid: {SAMPLE_INVENTORY_RELPATH}: "
            f"row {entry['row']}: {entry['reason']}"
            + (f" ({entry['sample']})" if entry["sample"] else "")
            for entry in invalid_rows]
        return {
            "status": "invalid",
            "note": f"{len(invalid_rows)} malformed inventory row(s) "
                    f"(reported, not filtered)",
            "invalid_rows": invalid_rows,
            "reasons": reasons,
        }

    expected_set = set(expected)

    suite_dirs = ({suite["dir"] for suite in discovery["python_suites"]}
                  | set(discovery["native_only_dirs"]))
    missing_samples = []
    missing_tests = []
    for sample in sorted(expected_set):
        if not (repo / "samples" / sample).is_dir():
            missing_samples.append(sample)
        elif f"samples/{sample}/tests" not in suite_dirs:
            missing_tests.append(sample)
    extra = []
    for domain in SAMPLE_DOMAINS:
        domain_dir = repo / "samples" / domain
        if not domain_dir.is_dir():
            continue
        for entry in sorted(domain_dir.iterdir()):
            if entry.is_dir() and f"{domain}/{entry.name}" not in expected_set:
                extra.append(f"{domain}/{entry.name}")

    result = {
        "status": "ok",
        "inventory": SAMPLE_INVENTORY_RELPATH,
        "expected_samples": len(expected_set),
        "missing_samples": missing_samples,
        "missing_tests_dirs": missing_tests,
        "extra_samples": extra,
    }
    if missing_samples or missing_tests or extra:
        result["status"] = "failed"
        reasons = []
        for sample in missing_samples:
            reasons.append(
                f"sample inventory: expected sample directory missing: "
                f"{sample}")
        for sample in missing_tests:
            reasons.append(
                f"sample inventory: {sample} has no discovered tests "
                f"directory")
        for sample in extra:
            reasons.append(
                f"sample inventory: sample directory not in inventory: "
                f"{sample}")
        result["reasons"] = reasons
    return result


# ---------------------------------------------------------------------------
# Worker: runs one unittest directory in-process and emits machine-readable
# JSON between sentinel markers on stdout.
# ---------------------------------------------------------------------------

class _JsonTestResult(unittest.TestResult):
    """TestResult that records non-ok outcomes with identity and reason."""

    def __init__(self):
        super().__init__()
        self.records = []
        self._current = None

    def startTest(self, test):
        super().startTest(test)
        self._current = {"id": test.id(), "outcome": "ok"}

    def stopTest(self, test):
        record = self._current or {"id": test.id(), "outcome": "ok"}
        if record["outcome"] != "ok":
            self.records.append(record)
        self._current = None
        super().stopTest(test)

    def _mark(self, test, outcome, err=None, reason=None):
        record = self._current
        if record is None or record["id"] != test.id():
            # Class-level events (setUpClass failures/skips) arrive on proxy
            # holders outside start/stopTest; keep them as standalone records.
            record = {"id": test.id(), "outcome": outcome}
            if reason is not None:
                record["reason"] = reason
            if err is not None:
                record["message"] = _format_error(err)
            self.records.append(record)
            return
        record["outcome"] = outcome
        if reason is not None:
            record["reason"] = reason
        if err is not None:
            record["message"] = _format_error(err)

    def addFailure(self, test, err):
        super().addFailure(test, err)
        self._mark(test, "fail", err=err)

    def addError(self, test, err):
        super().addError(test, err)
        self._mark(test, "error", err=err)

    def addSkip(self, test, reason):
        super().addSkip(test, reason)
        self._mark(test, "skip", reason=reason)

    def addExpectedFailure(self, test, err):
        super().addExpectedFailure(test, err)
        self._mark(test, "expected_failure")

    def addUnexpectedSuccess(self, test):
        super().addUnexpectedSuccess(test)
        self._mark(test, "unexpected_success")

    def addSubTest(self, test, subtest, err):
        super().addSubTest(test, subtest, err)
        if err is not None:
            self._mark(subtest, "error", err=err)


def _format_error(err):
    """Serialize either an exc_info tuple or a bare exception instance.

    ``unittest`` result hooks hand over ``sys.exc_info()`` tuples, while the
    worker's loader guard catches the exception object itself; both forms
    must serialize honestly (the bare form used to crash the worker with a
    ``TypeError`` instead of reporting the failure).
    """
    import traceback
    if isinstance(err, BaseException):
        text = "".join(traceback.format_exception(err))
    else:
        text = "".join(traceback.format_exception(*err))
    lines = text.splitlines()
    return "\n".join(lines[-40:])[:8000]


def _iter_tests(suite):
    for test in suite:
        if isinstance(test, unittest.TestSuite):
            yield from _iter_tests(test)
        else:
            yield test


def _import_failure_texts(suite):
    """Return the error text of loader-level (module import) failures."""
    texts = []
    for test in _iter_tests(suite):
        if type(test).__name__ != "_FailedTest":
            continue
        exception = getattr(test, "_exception", None)
        texts.append("" if exception is None else str(exception))
    return texts


def _emit(payload) -> int:
    sys.stdout.write(f"\n{WORKER_SENTINEL}\n")
    json.dump(payload, sys.stdout)
    sys.stdout.write("\n")
    sys.stdout.flush()
    return 0


def worker_main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="run.py _worker")
    parser.add_argument("--repo", required=True)
    parser.add_argument("--start-dir", required=True)
    args = parser.parse_args(argv)

    repo = Path(args.repo).resolve()
    start = Path(args.start_dir).resolve()
    sys.path.insert(0, str(repo))

    files = sorted(
        p.name for p in start.iterdir()
        if p.is_file() and p.name.startswith("test") and p.name.endswith(".py"))

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    optional_missing = []
    loader_errors = []
    for name in files:
        try:
            part = loader.discover(
                str(start), pattern=name, top_level_dir=str(start))
        except BaseException as exc:  # noqa: BLE001 - reported, never hidden
            loader_errors.append({"file": name, "error": _format_error(exc)})
            continue
        import_failures = _import_failure_texts(part)
        if import_failures:
            missing = missing_optional_dependency(import_failures[0])
            if missing and _is_export_scope(start / name):
                # Declared optional export scope: torch/funasr/ultralytics
                # module import.  Recorded, not executed, never a fake pass.
                optional_missing.append({"file": name, "missing": missing})
                continue
        for text in import_failures:
            loader_errors.append({"file": name, "error": text})
        suite.addTest(part)

    result = _JsonTestResult()
    captured_out, captured_err = io.StringIO(), io.StringIO()
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = captured_out, captured_err
    try:
        suite.run(result)
    finally:
        sys.stdout, sys.stderr = old_out, old_err

    payload = {
        "files": files,
        "tests_run": result.testsRun,
        "records": result.records,
        "optional_missing": optional_missing,
        "loader_errors": loader_errors,
        "captured_output": (captured_out.getvalue()
                            + captured_err.getvalue())[-16000:],
    }
    return _emit(payload)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class _SectionRunner:
    """Shared context for one orchestrator run."""

    def __init__(self, repo: Path, report: Path, python_exe: str,
                 options: dict):
        self.repo = repo.resolve()
        self.report = report.resolve()
        self.python = python_exe
        self.options = options
        self.artifacts = self.report.parent / "host-validation-artifacts"
        self.reasons = []

    # -- helpers ---------------------------------------------------------
    def log_file(self, name: str) -> Path:
        directory = self.artifacts / "logs"
        directory.mkdir(parents=True, exist_ok=True)
        return directory / name

    def run(self, command, *, cwd=None, timeout=None):
        return subprocess.run(
            command, cwd=None if cwd is None else str(cwd),
            capture_output=True, text=True, timeout=timeout)

    # -- python suites ---------------------------------------------------
    def run_python_suites(self, discovery):
        selected = discovery["python_suites"]
        filters = self.options["suite_filters"]
        if filters:
            selected = [s for s in selected
                        if any(f in s["dir"] for f in filters)]
        totals = {"tests": 0, "failures": 0, "errors": 0, "skipped": 0,
                  "optional_missing_modules": 0}
        suites_report = []
        allow_native = self.options["allow_native_skips"]
        for suite in selected:
            entry = self._run_one_suite(suite, totals, allow_native)
            suites_report.append(entry)
        missing = discovery["missing_directories"]
        for rel in missing:
            if filters and not any(f in rel for f in filters):
                continue
            self.reasons.append(f"missing declared test directory: {rel}")
        for anomaly in discovery["anomalies"]:
            self.reasons.append(f"unexplained test directory: {anomaly}")
        if filters and not selected and not missing:
            self.reasons.append(
                "suite filter matched nothing: " + ", ".join(filters))
        return {
            "discovered": len(discovery["python_suites"]),
            "selected": len(selected),
            "filtered_out": len(discovery["python_suites"]) - len(selected),
            "totals": totals,
            "suites": suites_report,
            "native_only_dirs": discovery["native_only_dirs"],
            "missing_directories": missing,
            "anomalies": discovery["anomalies"],
        }

    def _run_one_suite(self, suite, totals, allow_native):
        rel = suite["dir"]
        command = [
            self.python, str(RUNNER_PATH), "_worker",
            "--repo", str(self.repo),
            "--start-dir", str(self.repo / rel),
        ]
        started = time.monotonic()
        # Suite subprocesses execute at the requested repository: the
        # suites' README-command checks and their nested child
        # interpreters resolve repo-relative paths and repo-root imports
        # from the working directory, which must be the checkout — never
        # the caller's, since the maintainer may be launched from
        # anywhere.  Subprocess isolation, the outside-the-source report
        # rule and the caller's own cwd are unaffected; the worker's
        # --repo stays a path/sys.path identity input.
        try:
            proc = self.run(command, cwd=self.repo,
                            timeout=self.options["timeout_s"])
            timed_out = False
        except subprocess.TimeoutExpired as exc:
            proc = None
            timed_out = True
            stdout = exc.stdout.decode() if isinstance(exc.stdout, bytes) \
                else (exc.stdout or "")
            stderr = exc.stderr.decode() if isinstance(exc.stderr, bytes) \
                else (exc.stderr or "")
        duration = round(time.monotonic() - started, 3)
        entry = {
            "dir": rel,
            "command": command,
            "duration_s": duration,
            "status": "failed",
        }
        if timed_out:
            entry.update({
                "status": "timeout",
                "tests": 0, "failures": 0, "errors": 0, "skipped": 0,
                "skips": [], "optional_missing": [],
            })
            self.reasons.append(
                f"python suite {rel}: timeout after "
                f"{self.options['timeout_s']}s")
            totals["errors"] += 1
            return entry
        entry["exit_code"] = proc.returncode
        self.log_file(rel.replace("/", "__") + ".log").write_text(
            (proc.stdout or "") + "\n" + (proc.stderr or ""))
        payload = _parse_worker_output(proc.stdout)
        if payload is None or proc.returncode != 0:
            entry.update({
                "status": "worker-error",
                "stderr": (proc.stderr or "")[-4000:],
                "tests": 0, "failures": 0, "errors": 0, "skipped": 0,
                "skips": [], "optional_missing": [],
            })
            detail = "no result JSON" if payload is None else \
                f"exit {proc.returncode} after reporting results"
            self.reasons.append(
                f"python suite {rel}: worker error ({detail})")
            totals["errors"] += 1
            return entry

        failures = [r for r in payload["records"] if r["outcome"] == "fail"]
        errors = [r for r in payload["records"] if r["outcome"] == "error"]
        unexpected_success = [r for r in payload["records"]
                              if r["outcome"] == "unexpected_success"]
        skips = []
        rejected = []
        for record in payload["records"]:
            if record["outcome"] != "skip":
                continue
            category = classify_skip(record["id"], record.get("reason", ""),
                                     suite_dir=rel)
            item = {
                "id": record["id"],
                "reason": record.get("reason", ""),
                "category": category,
            }
            if category == "native_prerequisite" and allow_native:
                item["policy"] = "allowed-by-option"
            skips.append(item)
            if category == "native_prerequisite" and not allow_native:
                rejected.append((item, "native-prerequisite skip rejected in "
                                "strict mode"))
            elif category not in ("optional_export", "conditional",
                                  "native_prerequisite"):
                rejected.append((item, "undeclared skip"))

        tests = payload["tests_run"]
        loader_errors = payload["loader_errors"]
        entry.update({
            "tests": tests,
            "failures": len(failures) + len(unexpected_success),
            "errors": len(errors) + len(loader_errors),
            "skipped": len(skips),
            "skips": skips,
            "failures_detail": [
                {"id": r["id"], "message": r.get("message", "")}
                for r in failures + unexpected_success],
            "errors_detail": [
                {"id": r["id"], "message": r.get("message", "")}
                for r in errors],
            "loader_errors": loader_errors,
            "optional_missing": [
                {"file": item["file"], "missing": item["missing"]}
                for item in payload["optional_missing"]],
        })
        totals["tests"] += tests
        totals["failures"] += entry["failures"]
        totals["errors"] += entry["errors"]
        totals["skipped"] += len(skips)
        totals["optional_missing_modules"] += len(payload["optional_missing"])

        failed = False
        if entry["failures"] or entry["errors"]:
            self.reasons.append(
                f"python suite {rel}: {entry['failures']} failures, "
                f"{entry['errors']} errors")
            failed = True
        if tests == 0 and not payload["optional_missing"] and not failed:
            entry["status"] = "zero-tests"
            self.reasons.append(f"python suite {rel}: zero tests discovered")
            return entry
        for item, label in rejected:
            self.reasons.append(
                f"{label}: {rel}: {item['id']}: {item['reason']}")
            failed = True
        entry["status"] = "failed" if failed else "passed"
        return entry

    # -- static contract ---------------------------------------------------
    def run_contract(self):
        checker = self.repo / "utils/tools/sample_contract/check.py"
        if not checker.is_file():
            self.reasons.append(
                "static contract checker missing: utils/tools/sample_contract/check.py")
            return {"status": "missing-prerequisite"}
        report_file = self.artifacts / "sample-contract-report.json"
        report_file.parent.mkdir(parents=True, exist_ok=True)
        command = [
            self.python, str(checker), "--scope", "migration",
            "--parser-mode", "import", "--report", str(report_file),
        ]
        started = time.monotonic()
        try:
            proc = self.run(command, cwd=self.repo, timeout=3600)
        except subprocess.TimeoutExpired:
            self.reasons.append("static contract check timed out")
            return {"status": "timeout", "command": command}
        duration = round(time.monotonic() - started, 3)
        self.log_file("sample-contract.log").write_text(
            (proc.stdout or "") + "\n" + (proc.stderr or ""))
        section = {
            "command": command,
            "exit_code": proc.returncode,
            "duration_s": duration,
            "report_file": str(report_file),
        }
        summary = None
        if report_file.is_file():
            try:
                summary = json.loads(
                    report_file.read_text()).get("summary")
            except ValueError:
                summary = None
        section["summary"] = summary
        if proc.returncode != 0:
            section["status"] = "failed"
            self.reasons.append(
                f"static contract check failed (exit {proc.returncode})")
        else:
            section["status"] = "passed"
        return section

    # -- native prerequisites ---------------------------------------------
    def _pkg_config(self, module):
        """Probe a pkg-config module (best-effort, never raises)."""
        executable = shutil.which("pkg-config")
        if executable is None:
            return None
        try:
            proc = self.run(
                [executable, "--modversion", module], timeout=60)
        except (subprocess.SubprocessError, OSError):
            return None
        if proc.returncode != 0:
            return None
        return {"found": True, "detail": (proc.stdout or "").strip()}

    def check_native_prerequisites(self):
        section = {}
        failures = []

        def record(name, found, detail=""):
            section[name] = {"found": bool(found),
                             "detail": str(detail)[:2000]}
            if not found:
                failures.append(name)
            return found

        compiler = next(
            (path for path in
             (shutil.which(name) for name in ("c++", "g++", "clang++"))
             if path), None)
        if compiler:
            try:
                version = self.run([compiler, "--version"],
                                   timeout=60).stdout.splitlines()[0]
            except (subprocess.SubprocessError, IndexError):
                version = compiler
            record("compiler", True, version)
        else:
            record("compiler", False, "no c++/g++/clang++ on PATH")

        for tool, override in (("cmake", self.options["cmake"]),
                               ("ctest", self.options["ctest"])):
            executable = override or _find_tool(tool, self.python)
            record(tool, executable, executable or f"no {tool} on PATH")

        record("git", shutil.which("git"), shutil.which("git") or "")

        resolver_path = RUNNER_PATH.parent / "native_dependencies.py"
        resolver = None
        if resolver_path.is_file():
            try:
                spec = importlib.util.spec_from_file_location(
                    "host_validation_native_dependencies", resolver_path)
                resolver = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(resolver)
            except BaseException as exc:  # noqa: BLE001 - reported, not hidden
                record("resolver", False, f"import failed: {exc}")
        if resolver is None:
            if "resolver" not in section:
                record("resolver", False,
                       f"{resolver_path} not present next to run.py")
        else:
            section["resolver"] = {"found": True, "path": str(resolver_path)}
            for name, probe in (
                ("nlohmann_json", lambda: str(resolver.json_include_dir())),
                ("gflags", lambda: str(resolver.gflags_compile_flags())),
                ("iconv", lambda: str(resolver.iconv_link_flags())),
            ):
                try:
                    section[name] = {"found": True,
                                     "detail": probe()[:2000]}
                except Exception as exc:  # resolver's declared error types
                    section[name] = {"found": False, "detail": str(exc)[:2000]}
                    failures.append(name)

        section["pkg_config"] = {
            "found": bool(shutil.which("pkg-config")),
            "detail": shutil.which("pkg-config") or "",
        }
        # Frontend libraries for the default-on ASR_AUDIO_TESTS/ASR_CLI_TESTS
        # CTest flags.  Informational probes here; the hard gate is the CTest
        # configure result (find_path/find_library fail loudly when these
        # are absent), so the report always names the actionable install.
        section["ctest_frontend_libraries"] = {}
        for module, hint in (
            ("sndfile", "brew install libsndfile / "
                        "apt install libsndfile1-dev"),
            ("samplerate", "brew install libsamplerate / "
                           "apt install libsamplerate0-dev"),
        ):
            section["ctest_frontend_libraries"][module] = self._pkg_config(
                module) or {"found": False, "detail": hint}
        if failures:
            section["status"] = "missing"
            section["failures"] = failures
            if self.options["allow_native_skips"]:
                section["policy"] = "downgraded-by-option"
            else:
                for name in failures:
                    self.reasons.append(
                        f"native prerequisite missing: {name}: "
                        f"{section[name].get('detail', '')}")
                section["policy"] = "strict"
        else:
            section["status"] = "ok"
        return section

    # -- CTest projects ----------------------------------------------------
    def run_ctest(self):
        if self.options["skip_ctest"]:
            return {"status": "skipped-explicit", "projects": []}
        cmake = (self.options["cmake"] or _find_tool("cmake", self.python))
        ctest = (self.options["ctest"] or _find_tool("ctest", self.python))
        if cmake is None or ctest is None:
            self.reasons.append(
                "ctest section cannot run: cmake/ctest missing "
                "(native prerequisites are mandatory)")
            return {"status": "missing-prerequisite", "projects": []}
        section = {
            "default_defines": DEFAULT_CTEST_DEFINES,
            "mandatory_safe_defines": MANDATORY_SAFE_DEFINES,
        }
        if any(project.get("requires_opencv") for project in CTEST_PROJECTS):
            section["opencv_resolution"] = self.ctest_section_opencv()
        projects_report = []
        for project in CTEST_PROJECTS:
            projects_report.append(
                self._run_ctest_project(project, cmake, ctest))
        section.update({
            "status": "failed" if any(
                p["status"] != "passed" for p in projects_report) else "passed",
            # CTest case counts are separate on purpose: wrapper unittest
            # counts and native case counts must never be summed into one
            # "unique coverage" number.  The totals below are the actual
            # cases of the projects that ran — no hardcoded pass counts.
            "note": "CTest cases are counted separately from Python unittest "
                    "totals and are not additive with them.",
            "projects": projects_report,
        })
        return section

    def resolve_opencv(self, cmake):
        """Resolve the OpenCV CMake package robustly, without personal paths.

        A plain ``find_package(OpenCV)`` probe runs first (apt's
        ``libopencv-dev`` and Homebrew's default search resolve here).  If
        that fails, the documented ``OPENCV_DIR``/``OpenCV_DIR`` environment
        override wins, then standard package-manager prefixes — Homebrew
        installs the config as ``lib/cmake/opencv5``, which CMake does not
        find through the ``opencv4`` layout.  The result is recorded and the
        winning directory is passed as ``-DOpenCV_DIR`` to every
        OpenCV-requiring project.
        """
        probe_dir = self.artifacts / "opencv-probe"
        probe_dir.mkdir(parents=True, exist_ok=True)
        (probe_dir / "CMakeLists.txt").write_text(
            "cmake_minimum_required(VERSION 3.18)\n"
            "project(opencv_probe NONE)\n"
            "find_package(OpenCV QUIET COMPONENTS core imgproc imgcodecs)\n"
            "if(NOT OpenCV_FOUND)\n"
            "  message(FATAL_ERROR \"opencv not found\")\n"
            "endif()\n"
            "message(STATUS \"OPENCV_DIR=${OpenCV_DIR}\")\n")

        def probe(extra_args=()):
            try:
                proc = self.run(
                    [cmake, "-S", str(probe_dir),
                     "-B", str(probe_dir / "build"), *extra_args],
                    timeout=CTEST_STAGE_TIMEOUTS["configure"])
            except (subprocess.SubprocessError, OSError) as exc:
                return False, f"probe failed to run: {exc}"
            output = (proc.stdout or "") + "\n" + (proc.stderr or "")
            if proc.returncode != 0:
                return False, output.strip()[-500:]
            match = re.search(r"OPENCV_DIR=(\S+)", output)
            return True, (match.group(1) if match else "")

        ok, detail = probe()
        if ok:
            return {"mode": "default", "dir": detail or None,
                    "note": "plain find_package(OpenCV) resolved"}
        for override in ("OPENCV_DIR", "OpenCV_DIR"):
            value = os.environ.get(override)
            if value and (Path(value) / "OpenCVConfig.cmake").is_file():
                ok, detail = probe((f"-DOpenCV_DIR={value}",))
                if ok:
                    return {"mode": "env-override", "dir": value,
                            "override": override}
        for candidate in OPENCV_CONFIG_CANDIDATES:
            if not (Path(candidate) / "OpenCVConfig.cmake").is_file():
                continue
            ok, _ = probe((f"-DOpenCV_DIR={candidate}",))
            if ok:
                return {"mode": "prefix-probe", "dir": candidate}
        self.reasons.append(
            "OpenCV CMake package not found: neither find_package(OpenCV) "
            "nor the standard prefixes resolved it. Install libopencv-dev "
            "(Ubuntu) or opencv (Homebrew), or set OPENCV_DIR to its "
            "OpenCVConfig.cmake directory.")
        return {"mode": "not-found", "probe_output": detail}

    def _stage_timeout(self, stage):
        return int(self.options.get("ctest_stage_timeout_s")
                   or CTEST_STAGE_TIMEOUTS[stage])

    def _ctest_stage(self, stage, command, entry):
        """Run one CTest stage, containing timeout/OSError per project."""
        try:
            proc = self.run(command, timeout=self._stage_timeout(stage))
        except subprocess.TimeoutExpired as exc:
            def _text(value):
                if isinstance(value, bytes):
                    return value.decode(errors="replace")
                return value or ""
            entry["status"] = f"{stage}-timeout"
            entry["error"] = (f"timeout after {self._stage_timeout(stage)}s "
                              f"during {stage}")
            self.log_file(f'ctest-{entry["name"]}-{stage}.log').write_text(
                _text(exc.stdout) + "\n" + _text(exc.stderr)
                + f"\n[runner] {entry['error']}\n")
            return None
        except OSError as exc:
            entry["status"] = f"{stage}-error"
            entry["error"] = f"failed to execute {command[0]}: {exc}"
            self.log_file(f'ctest-{entry["name"]}-{stage}.log').write_text(
                f"[runner] {entry['error']}\n")
            return None
        self.log_file(f'ctest-{entry["name"]}-{stage}.log').write_text(
            (proc.stdout or "") + "\n" + (proc.stderr or ""))
        return proc

    def _run_ctest_project(self, project, cmake, ctest):
        name = project["name"]
        source = self.repo / project["source"]
        entry = {"name": name, "source": project["source"],
                 "note": project["note"], "status": "failed"}
        if not (source / "CMakeLists.txt").is_file():
            entry["status"] = "missing-project"
            self.reasons.append(
                f"registered CTest project missing: {project['source']}")
            return entry
        build = self.artifacts / "ctest" / name
        defines, _ = effective_cmake_defines(self.options)
        entry["defines"] = defines.get(name, {})
        command = [cmake, "-S", str(source), "-B", str(build)]
        for key, value in sorted(entry["defines"].items()):
            command.append(f"-D{key}={value}")
        if project.get("requires_opencv"):
            resolution = self.ctest_section_opencv()
            if resolution.get("dir") and resolution.get("mode") != "default":
                command.append(f"-DOpenCV_DIR={resolution['dir']}")
            entry["opencv"] = resolution.get("mode")
        started = time.monotonic()

        def finish(status, reason):
            entry["status"] = status
            entry["duration_s"] = round(time.monotonic() - started, 3)
            self.reasons.append(f"ctest project {name}: {reason}")
            return entry

        configure = self._ctest_stage("configure", command, entry)
        if configure is None:
            entry["duration_s"] = round(time.monotonic() - started, 3)
            self.reasons.append(
                f"ctest project {name}: cmake configure did not complete "
                f"({entry['status']})")
            return entry
        entry["configure_exit_code"] = configure.returncode
        if configure.returncode != 0:
            return finish(
                "configure-failed",
                f"cmake configure failed (exit {configure.returncode})")
        build_result = self._ctest_stage(
            "build",
            [cmake, "--build", str(build),
             "--parallel", str(os.cpu_count() or 2)], entry)
        if build_result is None:
            entry["duration_s"] = round(time.monotonic() - started, 3)
            self.reasons.append(
                f"ctest project {name}: build did not complete "
                f"({entry['status']})")
            return entry
        entry["build_exit_code"] = build_result.returncode
        if build_result.returncode != 0:
            return finish(
                "build-failed",
                f"build failed (exit {build_result.returncode})")
        declared = self._ctest_stage(
            "discover",
            [ctest, "--test-dir", str(build), "--show-only=json-v1"], entry)
        entry["declared_tests"] = None
        if declared is not None and declared.returncode == 0:
            try:
                entry["declared_tests"] = len(
                    json.loads(declared.stdout or "{}").get("tests", []))
            except ValueError:
                entry["declared_tests"] = None
        test_result = self._ctest_stage(
            "test",
            [ctest, "--test-dir", str(build), "--output-on-failure"], entry)
        if test_result is None:
            entry["duration_s"] = round(time.monotonic() - started, 3)
            self.reasons.append(
                f"ctest project {name}: test stage did not complete "
                f"({entry['status']})")
            return entry
        entry["test_exit_code"] = test_result.returncode
        summary = parse_ctest_output(test_result.stdout or "")
        entry["duration_s"] = round(time.monotonic() - started, 3)
        if summary is None:
            return finish("no-summary", "no CTest summary line parsed")
        entry.update(summary)
        if summary["total"] == 0:
            return finish("zero-tests", "zero CTest cases")
        if entry["declared_tests"] is None:
            return finish(
                "discovery-failed",
                "invalid or missing --show-only=json-v1 discovery output "
                "(declared case count unverifiable)")
        if entry["declared_tests"] != summary["total"]:
            return finish(
                "count-mismatch",
                f"declared {entry['declared_tests']} cases but "
                f"{summary['total']} ran")
        if test_result.returncode != 0 or summary["failed"] != 0:
            return finish(
                "failed",
                f"{summary['failed']} of {summary['total']} cases failed")
        entry["status"] = "passed"
        return entry

    def ctest_section_opencv(self):
        """Cached OpenCV resolution for this run (see resolve_opencv)."""
        if not hasattr(self, "_opencv_resolution_cache"):
            cmake = (self.options["cmake"]
                     or _find_tool("cmake", self.python))
            self._opencv_resolution_cache = (
                {"mode": "not-found"} if cmake is None
                else self.resolve_opencv(cmake))
        return self._opencv_resolution_cache

    # -- catalog -----------------------------------------------------------
    def run_catalog(self):
        if self.options["skip_catalog"]:
            return {"status": "skipped-explicit"}
        package = self.repo / "utils/tools/catalog-publisher"
        npm = shutil.which("npm")
        if npm is None:
            self.reasons.append("catalog check not run: npm not found")
            return {"status": "missing-prerequisite"}
        if not (package / "package.json").is_file():
            self.reasons.append(
                "catalog check not run: utils/tools/catalog-publisher/package.json "
                "missing")
            return {"status": "missing-prerequisite"}
        if not (package / "node_modules").is_dir():
            self.reasons.append(
                "catalog check not run: utils/tools/catalog-publisher/node_modules "
                "absent (run npm ci first; the runner never installs)")
            return {"status": "missing-prerequisite"}
        section = {"command": [npm, "run", "check"]}
        node = shutil.which("node")
        if node is None:
            self.reasons.append(
                "catalog engines check not run: node executable not found "
                "on PATH (npm shim without node cannot be verified)")
            return {**section, "status": "engines-missing"}
        # The declared Node range is enforced, not merely recorded: a run on
        # an unsupported Node can never be reported as a pass.
        try:
            engines = json.loads(
                (package / "package.json").read_text()).get("engines", {})
        except ValueError as exc:
            self.reasons.append(
                f"catalog engines check failed: package.json unparseable "
                f"({exc}); the supported Node range cannot be verified")
            return {**section, "status": "engines-invalid"}
        required = engines.get("node") if isinstance(engines, dict) else None
        if not required:
            self.reasons.append(
                "catalog engines check failed: package.json declares no "
                "engines.node range; add it (e.g. \">=22.12 <23\") so the "
                "supported Node version is verifiable")
            return {**section, "status": "engines-missing"}
        version = self.run([node, "--version"], timeout=60).stdout.strip()
        section["node"] = version
        section["engines"] = required
        section["engines_satisfied"] = _version_satisfies(
            version.lstrip("v"), required)
        if not section["engines_satisfied"]:
            self.reasons.append(
                f"Node {version} does not satisfy the catalog engines "
                f"declaration {required!r} (utils/tools/catalog-publisher/"
                f"package.json); install a Node in range (e.g. 22.12–22.x) "
                f"before running the catalog check")
            return {**section, "status": "engines-mismatch"}
        started = time.monotonic()
        try:
            proc = self.run(section["command"], cwd=package, timeout=3600)
        except subprocess.TimeoutExpired:
            self.reasons.append("catalog check timed out")
            return {**section, "status": "timeout"}
        except OSError as exc:
            self.reasons.append(f"catalog check failed to run: {exc}")
            return {**section, "status": "error"}
        section["exit_code"] = proc.returncode
        section["duration_s"] = round(time.monotonic() - started, 3)
        section["vitest"] = parse_vitest_summary(proc.stdout or "")
        self.log_file("catalog-npm-check.log").write_text(
            (proc.stdout or "") + "\n" + (proc.stderr or ""))
        if proc.returncode != 0:
            section["status"] = "failed"
            self.reasons.append(
                f"catalog check failed (exit {proc.returncode})")
        else:
            section["status"] = "passed"
        return section


def _interpreter_tool(tool):
    """Find ``tool`` next to the running interpreter (pip-provided cmake).

    The interpreter path is deliberately not resolved through symlinks: a
    virtualenv ``python`` is a symlink into the installation cellars, while
    the pip-provided ``cmake``/``ctest`` live in the venv's own bin dir.
    """
    candidate = Path(sys.executable).parent / tool
    if candidate.is_file() and os.access(candidate, os.X_OK):
        return str(candidate)
    return None


def _find_tool(tool, python_exe=None):
    """Locate a build tool: PATH, then the selected interpreter's bin dir.

    The interpreter lookup is deliberately not resolved through symlinks: a
    virtualenv ``python`` is a symlink into the installation cellars, while
    the pip-provided ``cmake``/``ctest`` live in the venv's own bin dir.
    """
    found = shutil.which(tool)
    if found:
        return found
    for executable in (python_exe, sys.executable):
        if not executable:
            continue
        candidate = Path(executable).parent / tool
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    return None


_CMAKE_TRUE = {"on", "true", "1", "yes"}

#: CMake ``if()`` false-constant semantics, verified against real CMake
#: 4.4.4 with an ``option()`` + ``if()`` probe (the ``independent-cmake-
#: false-constants`` evidence of the 2026-10-05 delivery-readiness round,
#: extended by the spelling/trailing-whitespace probe in
#: ``host_cmake_false_scope/execution/extra-cmake-aliases``): the named
#: constants ``0``, ``OFF``, ``NO``, ``FALSE``, ``N``, ``IGNORE`` and the
#: empty value compare case-insensitively; ``NOTFOUND`` — the exact token
#: and the ``-NOTFOUND`` suffix — is matched case-sensitively (``notfound``
#: and ``X-notfound`` are truthy); and CMake strips trailing whitespace
#: from a ``-D`` value when caching it while keeping leading whitespace
#: (``"OFF "`` is really OFF, ``" OFF"`` and ``" OFF "`` stay ON).
_CMAKE_FALSE = {"", "0", "off", "no", "false", "n", "ignore"}


def _is_cmake_false(value):
    """Real-CMake ``if()`` false-constant test for a raw ``-D`` value.

    Classification mirrors what CMake actually evaluates: the value with
    trailing whitespace stripped (as CMake caches it), leading whitespace
    preserved, named constants case-insensitive, ``NOTFOUND`` and the
    ``-NOTFOUND`` suffix case-sensitive.
    """
    token = value.rstrip()
    if token.endswith("-NOTFOUND") or token == "NOTFOUND":
        return True
    return token.lower() in _CMAKE_FALSE


def _is_exact_cmake_false(value):
    """Exact, unpadded CMake false-constant spelling.

    The mandated-OFF guard accepts a restatement only in this form: any
    surrounding whitespace makes the spelling fail-closed rejected (CMake
    keeps leading whitespace, and this gate never guesses which padding
    CMake might strip), even where the stripped token is a false constant.
    """
    return value == value.strip() and _is_cmake_false(value)


def effective_cmake_defines(options):
    """Merge default full-gate defines with ``--cmake-define`` overrides.

    Returns ``(merged, reductions)`` where ``reductions`` names every user
    define that turned a default-on flag off — a scope reduction that makes
    the run non-CI-equivalent.  Classification matches on the bare key
    name: a typed spelling (``VAR:BOOL`` — rejected by the CLI parser, but
    classified conservatively here regardless) whose base name targets a
    default-ON flag is recorded as a reduction too, so no define spelling
    can turn required test/IO/sanitizer flags off while the run still
    claims CI equivalence.  The typed key itself is kept as given; this
    function never rewrites defines into supported bare spellings.  The
    value is classified the way real CMake evaluates it
    (:func:`_is_cmake_false`; verified against CMake 4.4.4): a false
    constant — ``0``/``OFF``/``NO``/``FALSE``/``N``/``IGNORE`` and the
    empty value case-insensitively, or the case-sensitive exact ``NOTFOUND``
    and ``-NOTFOUND`` suffix — turns the flag off in fact and is recorded
    as a reduction naming the raw spelling it arrived as.  CMake strips
    trailing whitespace from a ``-D`` value when caching it, so a
    trailing-padded false token is a real OFF and a recorded reduction
    too, while leading whitespace survives (``" OFF"`` and ``" OFF "``
    keep the flag ON in fact and are correctly *not* recorded reductions —
    the guard for mandated-OFF switches rejects padded spellings outright
    before this merge ever runs).
    """
    merged = {name: dict(defines)
              for name, defines in DEFAULT_CTEST_DEFINES.items()}
    reductions = []
    for project, defines in options.get("cmake_defines", {}).items():
        for key, value in defines.items():
            base = key.split(":", 1)[0]
            default = DEFAULT_CTEST_DEFINES.get(project, {}).get(base)
            if (default is not None
                    and default.lower() in _CMAKE_TRUE
                    and _is_cmake_false(value)):
                reductions.append(f"{project}:{key}={value}")
            merged.setdefault(project, {})[key] = value
    return merged, reductions


def prohibited_cmake_overrides(options):
    """Return rejection reasons for defines that would enable a vendor build.

    The switches in :data:`MANDATORY_SAFE_DEFINES` keep the vendor SDK
    adapters and production CLIs of the registered runtime/cpp projects
    OFF.  A ``--cmake-define`` that enables one is rejected before any
    project is configured — this maintainer host tool never constructs
    those builds, so an enabling override is neither merged nor recorded
    as a scope change.  Restating the mandated OFF value is a no-op.  The
    switch is matched on its bare key name, so a typed spelling
    (``VAR:BOOL`` — already rejected by the CLI parser) can never slip an
    enabling value past the exact-key comparison either.  The value is
    classified with the same CMake false-constant semantics as the scope
    accounting (:func:`_is_cmake_false`), but the accepted no-op set is
    deliberately stricter — only an exact, unpadded false-constant
    spelling (:func:`_is_exact_cmake_false`) restates the mandated OFF
    value.  Every padded spelling is rejected fail-closed: CMake keeps
    leading whitespace on ``-D`` values (``" OFF"`` leaves the switch
    enabled in fact) and this gate never guesses which padding CMake
    might strip, so no padded false-looking token is trusted — and the
    case-sensitive ``NOTFOUND``/``-NOTFOUND`` spellings aside, the named
    constants compare case-insensitively, so no casing of an enabling
    value slips through as a supposed restatement.
    """
    reasons = []
    for project in sorted(options.get("cmake_defines", {})):
        defines = options["cmake_defines"][project]
        safe = MANDATORY_SAFE_DEFINES.get(project, {})
        for key in sorted(defines):
            value = defines[key]
            base = key.split(":", 1)[0]
            if base not in safe:
                continue
            if _is_exact_cmake_false(value):
                continue  # restating the mandated OFF value: a no-op
            typed_note = ("" if base == key else
                          f"; the typed spelling {key} is not supported by "
                          f"--cmake-define (untyped {base} only)")
            padded_note = (
                "" if (value == value.strip()
                       or not _is_cmake_false(value.strip())) else
                "; the surrounding whitespace is never trimmed away to "
                "rescue the value (CMake keeps leading whitespace, so a "
                "padded false-looking token cannot be relied on as OFF) "
                "— pass the exact token")
            reasons.append(
                f"{project}:{key}={value}: rejected — {base} is a "
                f"vendor/production build switch this host gate keeps "
                f"{safe[base]}; the maintainer host tool never builds the "
                f"vendor SDK or the production CLI, so it cannot be enabled "
                f"through --cmake-define{typed_note}{padded_note}")
    return reasons


def _parse_worker_output(stdout):
    if not stdout:
        return None
    marker = stdout.rfind(WORKER_SENTINEL)
    if marker == -1:
        return None
    tail = stdout[marker + len(WORKER_SENTINEL):].strip()
    try:
        payload = json.loads(tail)
    except ValueError:
        return None
    if not isinstance(payload, dict):
        return None
    for key in ("files", "tests_run", "records", "optional_missing",
                "loader_errors"):
        if key not in payload:
            return None
    return payload


def _version_satisfies(version, range_text):
    """Minimal semver-range check supporting the engines strings we declare."""
    parts = []
    for piece in re.findall(r"\d+", version):
        parts.append(int(piece))
    for constraint in range_text.split():
        match = re.fullmatch(r"(>=|<=|>|<|==|=)?([\d.*]+)", constraint)
        if match is None:
            return False
        operator = match.group(1) or "=="
        target = [int(p) if p.isdigit() else 0
                  for p in match.group(2).split(".")]
        while len(target) < len(parts):
            target.append(0)
        while len(parts) < len(target):
            parts.append(0)
        if operator == ">=" and not parts >= target:
            return False
        if operator == "<=" and not parts <= target:
            return False
        if operator == ">" and not parts > target:
            return False
        if operator == "<" and not parts < target:
            return False
        if operator in ("==", "=") and parts != target:
            return False
    return True


def _sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run.py",
        description="Maintainer host-validation runner (not a user CLI).")
    parser.add_argument("--repo", required=True, type=Path,
                        help="repository checkout to validate")
    parser.add_argument("--report", type=Path,
                        help="machine-readable report path (required unless "
                             "--list)")
    parser.add_argument("--python", default=sys.executable,
                        help="interpreter used for the test suites "
                             "(default: this interpreter)")
    parser.add_argument("--timeout", type=int, default=1800,
                        help="per-suite timeout in seconds (default 1800)")
    parser.add_argument("--suite", action="append", default=[],
                        metavar="SUBSTRING",
                        help="run only suites whose directory contains "
                             "SUBSTRING (repeatable; not CI-equivalent)")
    parser.add_argument("--list", action="store_true",
                        help="print discovery as JSON and exit")
    parser.add_argument("--skip-ctest", action="store_true",
                        help="skip the native CTest section (explicitly "
                             "recorded; not CI-equivalent)")
    parser.add_argument("--skip-contract", action="store_true",
                        help="skip the static contract section (explicitly "
                             "recorded; not CI-equivalent)")
    parser.add_argument("--skip-catalog", action="store_true",
                        help="skip the catalog section (explicitly recorded; "
                             "not CI-equivalent); no dist/catalog.json is "
                             "built then, so the ultralytics_yolo asset/"
                             "manifest snapshot suites need a previously "
                             "generated catalog or they fail")
    parser.add_argument("--allow-native-skips", action="store_true",
                        help="record native-prerequisite skips instead of "
                             "rejecting them (not CI-equivalent)")
    parser.add_argument("--cmake", help="cmake executable override")
    parser.add_argument("--ctest", help="ctest executable override")
    parser.add_argument("--cmake-define", action="append", default=[],
                        metavar="PROJECT:VAR=VALUE",
                        help="override a -D define for one CTest project "
                             "(repeatable); the full-gate defaults "
                             "(YOLOE_TEST_OPENCV/ASR_AUDIO_TESTS/"
                             "ASR_CLI_TESTS=ON, and for the runtime/cpp "
                             "projects PARAFORMER_BUILD_TESTS/IO + "
                             "PARAFORMER_SANITIZERS, HIMLOCO_BUILD_TESTS=ON) "
                             "are merged in — turning a default-ON flag off "
                             "via any CMake false constant is a recorded "
                             "scope reduction; the vendor SDK/"
                             "production CLI switches (PARAFORMER_BUILD_SDK/"
                             "BUILD_CLI, HIMLOCO_BUILD_SDK/BUILD_CLI) are "
                             "mandated OFF and enabling overrides are "
                             "rejected (restating an exact false constant — "
                             "0/OFF/NO/FALSE/N/IGNORE case-insensitively, "
                             "NOTFOUND or a *-NOTFOUND value case-sensitively "
                             "— is a no-op; CMake keeps leading whitespace "
                             "on -D values, so whitespace-padded "
                             "false-looking tokens are rejected too); only "
                             "the untyped "
                             "PROJECT:VAR=VALUE spelling is accepted — typed "
                             "CMake cache keys (VAR:BOOL, VAR:STRING, "
                             "VAR:PATH, ...) are unsupported and rejected "
                             "before any configure/build")
    return parser


def _parse_cmake_defines(raw):
    """Parse ``--cmake-define`` values: ``PROJECT:VAR=VALUE`` only.

    Typed CMake cache keys (``VAR:BOOL``, ``VAR:STRING``, ``VAR:PATH``,
    ``VAR:FILEPATH``, ``VAR:INTERNAL``, ...) are explicitly rejected: CMake
    lets a later typed define override an earlier bare one, so a typed key
    is exactly how a mandated-OFF vendor switch could be re-enabled past
    the exact-key guard or a required default-ON flag turned off without a
    recorded scope reduction.  This CLI does not interpret cache types —
    rejecting the spelling outright, here (before anything is configured
    or built), is the supported contract.
    """
    defines = {}
    for item in raw:
        if ":" not in item or "=" not in item:
            raise SystemExit(f"invalid --cmake-define (want PROJECT:VAR=VALUE): {item}")
        project, assignment = item.split(":", 1)
        key, value = assignment.split("=", 1)
        if ":" in key:
            raise SystemExit(
                f"invalid --cmake-define (typed CMake cache keys are not "
                f"supported): {item}\n"
                f"  pass the untyped spelling PROJECT:VAR=VALUE instead "
                f"(e.g. --cmake-define {project}:{key.split(':', 1)[0]}"
                f"={value}); typed keys are rejected before any "
                f"configure/build because a later VAR:TYPE define would "
                f"override the mandated-OFF vendor SDK/CLI switches and "
                f"bypass the recorded scope accounting")
        defines.setdefault(project, {})[key] = value
    return defines


def main(argv=None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    if argv and argv[0] == "_worker":
        return worker_main(argv[1:])
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.report is None and not args.list:
        parser.error("--report is required unless --list is given")

    repo = args.repo.resolve()
    if not repo.is_dir():
        print(f"repository not found: {repo}", file=sys.stderr)
        return 2

    discovery = discover_python_suites(repo)
    if args.list:
        print(json.dumps({
            "python_suites": discovery["python_suites"],
            "native_only_dirs": discovery["native_only_dirs"],
            "anomalies": discovery["anomalies"],
            "missing_directories": discovery["missing_directories"],
            "sample_coverage": validate_sample_coverage(repo, discovery),
            "pins": declared_pins(repo),
            "ctest_projects": [
                {k: v for k, v in project.items() if k != "note"}
                for project in CTEST_PROJECTS],
            "ctest_default_defines": DEFAULT_CTEST_DEFINES,
            "ctest_mandatory_safe_defines": MANDATORY_SAFE_DEFINES,
        }, indent=2))
        return 0

    report = args.report.resolve()
    # Reports and artifacts must live outside the source tree: a report
    # written inside it would both pollute the worktree and defeat the
    # content-drift gate (self-induced dirty state).
    try:
        report.relative_to(repo)
    except ValueError:
        pass  # report outside the repository: required
    else:
        print(
            f"--report must live outside the repository checkout "
            f"(got {report} inside {repo}); reports and artifacts next to it "
            f"must never be written into the source tree",
            file=sys.stderr)
        return 2
    report.parent.mkdir(parents=True, exist_ok=True)
    options = {
        "suite_filters": args.suite,
        "timeout_s": args.timeout,
        "allow_native_skips": args.allow_native_skips,
        "skip_ctest": args.skip_ctest,
        "skip_catalog": args.skip_catalog,
        "skipped_sections": sorted(
            name for flag, name in ((args.skip_ctest, "ctest"),
                                    (args.skip_contract, "contract"),
                                    (args.skip_catalog, "catalog")) if flag),
        "cmake": args.cmake,
        "ctest": args.ctest,
        "cmake_defines": _parse_cmake_defines(args.cmake_define),
    }
    # Vendor SDK / production CLI switches are mandated OFF: reject an
    # enabling --cmake-define before anything is configured or reported —
    # the host tool never constructs those builds.
    prohibited = prohibited_cmake_overrides(options)
    if prohibited:
        for reason in prohibited:
            print(f"host-validation: rejected --cmake-define: {reason}",
                  file=sys.stderr)
        return 2
    _, reductions = effective_cmake_defines(options)
    options["cmake_scope_reductions"] = reductions
    options["ci_equivalent"] = not (
        options["skipped_sections"] or options["allow_native_skips"]
        or options["suite_filters"] or reductions)

    python_version = subprocess.run(
        [args.python, "--version"], capture_output=True, text=True)
    started_utc = datetime.now(timezone.utc).isoformat()
    started = time.monotonic()

    snapshot_start = git_snapshot(repo)
    manifest_start = source_manifest(repo)
    pins = verify_pins(repo)

    runner_section = _SectionRunner(repo, report, args.python, options)
    runner_section.artifacts.mkdir(parents=True, exist_ok=True)

    # The catalog gate runs before the Python suites: its ``npm run check``
    # build stage generates ``utils/tools/catalog-publisher/dist/catalog.json``
    # (ignored build output), which the ultralytics_yolo asset/manifest
    # snapshot suites read — on a clean checkout no ``dist/`` exists, so
    # building the catalog after those suites fails them.  It still runs
    # after the starting source snapshots and before the ending ones, i.e.
    # inside the same source-stability interval, so the ordering never
    # weakens the content gate.
    catalog_section = runner_section.run_catalog()
    python_section = runner_section.run_python_suites(discovery)
    coverage_section = validate_sample_coverage(repo, discovery)
    if coverage_section["status"] != "ok":
        # The accepted inventory is required proof: not-present, invalid
        # and failed all carry explicit structured reasons that fail the
        # run — a missing or malformed inventory can never silently
        # disable the coverage gate or claim full CI equivalence.
        for reason in coverage_section["reasons"]:
            runner_section.reasons.append(reason)
    native_section = runner_section.check_native_prerequisites()
    contract_section = ({"status": "skipped-explicit"}
                        if args.skip_contract
                        else runner_section.run_contract())
    if contract_section.get("status") == "skipped-explicit":
        options["skipped_sections"] = sorted(
            set(options["skipped_sections"]) | {"contract"})
    ctest_section = runner_section.run_ctest()

    for pin in pins:
        if not pin["present"]:
            runner_section.reasons.append(
                f"pinned commit {pin['pin']} (from {pin['source']}) is "
                f"missing from this clone: git fetch origin {pin['pin']}")

    snapshot_end = git_snapshot(repo)
    manifest_end = source_manifest(repo)
    source_drift = (
        snapshot_start["commit"] != snapshot_end["commit"])
    dirty_changed = (
        snapshot_start["dirty_files"] != snapshot_end["dirty_files"])
    content_drift = (
        manifest_start is None or manifest_end is None
        or manifest_start["digest"] != manifest_end["digest"])
    if source_drift:
        runner_section.reasons.append(
            "source drift: HEAD changed during the run "
            f"({snapshot_start['commit']} -> {snapshot_end['commit']})")
    if dirty_changed:
        runner_section.reasons.append(
            "source drift: the dirty-file list changed during the run")
    if manifest_start is None or manifest_end is None:
        runner_section.reasons.append(
            "source identity unavailable: not a Git repository (or git "
            "missing/failed); a strict gate cannot certify this checkout")
    elif content_drift:
        runner_section.reasons.append(
            "source content changed during the run (content digest "
            f"{manifest_start['digest'][:12]}.. -> "
            f"{manifest_end['digest'][:12]}..)")

    reasons = runner_section.reasons
    document = {
        "schema": REPORT_SCHEMA,
        "runner": {
            "path": str(RUNNER_PATH),
            "sha256": _sha256_of(RUNNER_PATH),
            "python": args.python,
            "python_version": (python_version.stdout
                               or python_version.stderr).strip(),
            "argv": [str(a) for a in (argv if argv is not None else sys.argv[1:])],
        },
        "host": {
            "system": host_platform.system(),
            "release": host_platform.release(),
            "machine": host_platform.machine(),
        },
        "started_utc": started_utc,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "duration_s": round(time.monotonic() - started, 3),
        "options": options,
        "source": {
            "repo": str(repo),
            "commit": snapshot_end["commit"],
            "branch": snapshot_end["branch"],
            "start_commit": snapshot_start["commit"],
            "start_branch": snapshot_start["branch"],
            "dirty_files_start": snapshot_start["dirty_files"],
            "dirty_files_end": snapshot_end["dirty_files"],
            "dirty_count_start": snapshot_start["dirty_count"],
            "dirty_count_end": snapshot_end["dirty_count"],
            "manifest": None if manifest_start is None else {
                "algorithm": manifest_start["algorithm"],
                "digest_start": manifest_start["digest"],
                "digest_end":
                    None if manifest_end is None else manifest_end["digest"],
                "file_count_start": manifest_start["file_count"],
                "file_count_end":
                    None if manifest_end is None else manifest_end["file_count"],
                "gitlinks": manifest_start["gitlinks"],
                "files_hashed_sample":
                    manifest_start["files_hashed_sample"],
            },
            "source_drift": source_drift,
            "content_drift": content_drift,
            "dirty_files_changed_during_run": dirty_changed,
            "pins": pins,
        },
        "native_prerequisites": native_section,
        "python_suites": python_section,
        "sample_coverage": coverage_section,
        "contract": contract_section,
        "ctest": ctest_section,
        "catalog": catalog_section,
        "overall": {
            "status": "failed" if reasons else "passed",
            "reasons": reasons,
            "totals": {
                "python_tests": python_section["totals"]["tests"],
                "python_failures": python_section["totals"]["failures"],
                "python_errors": python_section["totals"]["errors"],
                "python_skipped": python_section["totals"]["skipped"],
                "python_optional_missing_modules":
                    python_section["totals"]["optional_missing_modules"],
                "ctest_cases": (
                    sum(p.get("total", 0)
                        for p in ctest_section.get("projects", []))
                    if isinstance(ctest_section.get("projects"), list)
                    else 0),
                "counting_note": "ctest cases are separate from python tests",
            },
        },
    }
    report.write_text(json.dumps(document, indent=2))

    failed_suites = [
        s for s in python_section["suites"] if s["status"] != "passed"]
    print(f"host-validation: {document['overall']['status']} "
          f"({document['duration_s']}s)")
    print(f"  python suites: {python_section['selected']}/"
          f"{python_section['discovered']} selected, "
          f"{python_section['totals']['tests']} tests, "
          f"{len(failed_suites)} failing suites")
    print(f"  contract: {contract_section.get('status')}  "
          f"ctest: {ctest_section.get('status')}  "
          f"catalog: {catalog_section.get('status')}  "
          f"sample coverage: {coverage_section.get('status')}  "
          f"native prerequisites: {native_section.get('status')}")
    for reason in reasons:
        print(f"  reason: {reason}")
    print(f"  report: {report}")
    print(f"  artifacts: {runner_section.artifacts}")
    return 1 if reasons else 0


if __name__ == "__main__":
    sys.exit(main())
