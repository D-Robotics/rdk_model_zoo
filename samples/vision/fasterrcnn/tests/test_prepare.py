# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Regressions for the model preparation entry (2026-10-13).

``model/prepare.py --from-build`` must be idempotent: a half whose source
already is the installed file (``--from-build model/<target>/`` with the
flat pair layout), or whose bytes are already installed, is never copied,
and a differing install is refused without ``--force``.

The tool must also run on a plain host: the quickstart compiles inside
Docker but installs and verifies the artifacts outside it, so importing
``prepare.py`` and hashing files must never pull the runtime's NumPy /
OpenCV dependency chain.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import json
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PREPARE_PATH = ROOT / "samples" / "vision" / "fasterrcnn" / "model" / "prepare.py"


def load_prepare():
    """Import model/prepare.py (a script, not a package module)."""

    spec = importlib.util.spec_from_file_location(
        "fasterrcnn_model_prepare", PREPARE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


PREPARE = load_prepare()

STALE_TIME = 1_000_000


class InstallPairIdempotencyTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.model_dir = Path(self._tmp.name) / "model"
        self._orig_model_dir = PREPARE.MODEL_DIR
        PREPARE.MODEL_DIR = self.model_dir
        self.addCleanup(setattr, PREPARE, "MODEL_DIR", self._orig_model_dir)
        self.pair = SimpleNamespace(
            parta_file="parta.bin", partb_file="partb.bin",
            parta_sha256="0" * 64, partb_sha256="0" * 64)

    def _install(self, parta_src: Path, partb_src: Path, *,
                 force: bool = False, dry_run: bool = False):
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            ok = PREPARE.install_pair("x5", self.pair, parta_src, partb_src,
                                      force=force, dry_run=dry_run)
        return ok, stdout.getvalue()

    @staticmethod
    def _identity(path: Path):
        stat = path.stat()
        return (stat.st_ino, stat.st_mtime_ns)

    def test_same_path_source_is_left_in_place(self):
        # --from-build model/x5 with the flat pair there: src == dst, which
        # shutil.copy2 turns into SameFileError.
        dst_dir = self.model_dir / "x5"
        dst_dir.mkdir(parents=True)
        parta = dst_dir / "parta.bin"
        partb = dst_dir / "partb.bin"
        parta.write_bytes(b"parta-bytes")
        partb.write_bytes(b"partb-bytes")
        before = (self._identity(parta), self._identity(partb))

        ok, output = self._install(parta, partb)

        self.assertTrue(ok)
        self.assertIn("already the installed file", output)
        self.assertEqual((self._identity(parta), self._identity(partb)), before)

    def test_identical_bytes_are_not_recopied(self):
        build = Path(self._tmp.name) / "build"
        build.mkdir()
        (build / "parta.bin").write_bytes(b"same-bytes-a")
        (build / "partb.bin").write_bytes(b"same-bytes-b")
        dst_dir = self.model_dir / "x5"
        dst_dir.mkdir(parents=True)
        parta = dst_dir / "parta.bin"
        partb = dst_dir / "partb.bin"
        parta.write_bytes(b"same-bytes-a")
        partb.write_bytes(b"same-bytes-b")
        # A stale timestamp proves the file was left in place, not rewritten.
        os.utime(parta, (STALE_TIME, STALE_TIME))
        os.utime(partb, (STALE_TIME, STALE_TIME))

        ok, output = self._install(build / "parta.bin", build / "partb.bin")

        self.assertTrue(ok)
        self.assertIn("left in place", output)
        self.assertEqual(parta.stat().st_mtime_ns, STALE_TIME * 1_000_000_000)
        self.assertEqual(partb.stat().st_mtime_ns, STALE_TIME * 1_000_000_000)

    def test_differing_install_refused_without_force(self):
        build = Path(self._tmp.name) / "build"
        build.mkdir()
        (build / "parta.bin").write_bytes(b"new-a")
        (build / "partb.bin").write_bytes(b"new-b")
        dst_dir = self.model_dir / "x5"
        dst_dir.mkdir(parents=True)
        (dst_dir / "parta.bin").write_bytes(b"old-a")
        (dst_dir / "partb.bin").write_bytes(b"old-b")

        ok, output = self._install(build / "parta.bin", build / "partb.bin")

        self.assertFalse(ok)
        self.assertIn("refusing to overwrite", output)
        self.assertEqual((dst_dir / "parta.bin").read_bytes(), b"old-a")
        self.assertEqual((dst_dir / "partb.bin").read_bytes(), b"old-b")

    def test_force_replaces_differing_install(self):
        build = Path(self._tmp.name) / "build"
        build.mkdir()
        (build / "parta.bin").write_bytes(b"new-a")
        (build / "partb.bin").write_bytes(b"new-b")
        dst_dir = self.model_dir / "x5"
        dst_dir.mkdir(parents=True)
        (dst_dir / "parta.bin").write_bytes(b"old-a")
        (dst_dir / "partb.bin").write_bytes(b"old-b")

        ok, output = self._install(build / "parta.bin", build / "partb.bin",
                                   force=True)

        self.assertTrue(ok)
        self.assertEqual((dst_dir / "parta.bin").read_bytes(), b"new-a")
        self.assertEqual((dst_dir / "partb.bin").read_bytes(), b"new-b")

    def test_dry_run_reports_without_writing(self):
        build = Path(self._tmp.name) / "build"
        build.mkdir()
        (build / "parta.bin").write_bytes(b"dry-a")
        (build / "partb.bin").write_bytes(b"dry-b")

        ok, output = self._install(build / "parta.bin", build / "partb.bin",
                                   dry_run=True)

        self.assertTrue(ok)
        self.assertIn("would install", output)
        self.assertFalse((self.model_dir / "x5").exists())


class PrepareMainIdempotencyTests(unittest.TestCase):
    """End-to-end ``--from-build`` runs through main() with the real descriptor."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.model_dir = Path(self._tmp.name) / "model"
        self._orig_model_dir = PREPARE.MODEL_DIR
        PREPARE.MODEL_DIR = self.model_dir
        self.addCleanup(setattr, PREPARE, "MODEL_DIR", self._orig_model_dir)
        self.pair = PREPARE.descriptor_pairs()["x5"]

    def _main(self, *argv):
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            code = PREPARE.main(list(argv))
        return code, stdout.getvalue()

    def test_rerun_and_install_from_model_dir_are_idempotent(self):
        build = Path(self._tmp.name) / "build" / "x5"
        build.mkdir(parents=True)
        (build / self.pair.parta_file).write_bytes(b"host-fake-part-a")
        (build / self.pair.partb_file).write_bytes(b"host-fake-part-b")

        # First install copies; the fake bytes differ from the recorded
        # descriptor digests, so the run reports exit code 2.
        code, output = self._main("--target", "x5",
                                  "--from-build", str(build.parent))
        self.assertEqual(code, 2)
        installed = self.model_dir / "x5"
        parta = installed / self.pair.parta_file
        partb = installed / self.pair.partb_file
        identity = (parta.stat().st_ino, partb.stat().st_ino)

        # A second run from the same build directory must not rewrite.
        code, output = self._main("--target", "x5",
                                  "--from-build", str(build.parent))
        self.assertEqual(code, 2)
        self.assertIn("left in place", output)
        self.assertEqual((parta.stat().st_ino, partb.stat().st_ino), identity)

        # Installing from model/x5 itself (flat single-target pair) must
        # keep the files in place instead of raising SameFileError.
        code, output = self._main("--target", "x5",
                                  "--from-build", str(installed))
        self.assertEqual(code, 2)
        self.assertIn("already the installed file", output)
        self.assertEqual((parta.stat().st_ino, partb.stat().st_ino), identity)


class StdlibOnlyPrepareTests(unittest.TestCase):
    """``prepare.py`` imports and hashes with no third-party modules.

    The Codex repro: ``python3 -S`` + ``importlib`` load of ``prepare.py``
    used to fail with ``ModuleNotFoundError: numpy`` because
    ``sha256_file`` imported the runtime's ``detect.file_sha256``.  The
    subprocess below runs with ``-S`` (no site-packages: any third-party
    import fails outright), imports the tool, hashes a real file larger
    than one read chunk, and reports which heavy modules landed in
    ``sys.modules``.
    """

    _PROBE = """
        import importlib.util, json, sys
        from pathlib import Path

        spec = importlib.util.spec_from_file_location(
            "prepare_under_s", sys.argv[1])
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        digest = module.sha256_file(Path(sys.argv[2]))
        heavy = ("numpy", "cv2", "torch", "torchvision", "onnx",
                 "onnxruntime", "PIL")
        print(json.dumps({
            "digest": digest,
            "heavy_modules": sorted(
                name for name in sys.modules
                if name.split(".")[0] in heavy),
        }))
    """

    def test_import_and_hash_under_sys_flag_s(self):
        payload = bytes(range(256)) * 4097  # 1 MiB + 256 B: two read chunks
        with tempfile.TemporaryDirectory() as tmp:
            artifact = Path(tmp) / "artifact.bin"
            artifact.write_bytes(payload)
            proc = subprocess.run(
                [sys.executable, "-S", "-c", textwrap.dedent(self._PROBE),
                 str(PREPARE_PATH), str(artifact)],
                capture_output=True, text=True)

        self.assertEqual(
            proc.returncode, 0,
            f"prepare.py is not plain-host importable:\n{proc.stderr}")
        result = json.loads(proc.stdout)
        self.assertEqual(result["heavy_modules"], [])
        self.assertEqual(result["digest"],
                         hashlib.sha256(payload).hexdigest())


if __name__ == "__main__":
    unittest.main()
