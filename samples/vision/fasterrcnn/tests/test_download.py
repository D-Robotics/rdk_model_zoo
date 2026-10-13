# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Regressions for the published-pair download entry (2026-10-13).

``model/download.py`` turns the release-manifest rows into the runtime's
default PartA/PartB lookup paths.  These tests pin, per target: the exact
manifest references and public archive URLs, agreement between the manifest
digests and ``model/frcnn_mbv3_320.json`` (the two registries must never
drift apart), the destination layout, the exit-2 failure path, and — through
the real shared downloader with a stubbed HTTP source — that only verified
bytes are ever installed and a mismatch leaves no file behind.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DOWNLOAD_PATH = ROOT / "samples" / "vision" / "fasterrcnn" / "model" / "download.py"


def load_download():
    """Import model/download.py (a script, not a package module)."""

    spec = importlib.util.spec_from_file_location(
        "fasterrcnn_model_download", DOWNLOAD_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


DOWNLOAD = load_download()

ARCHIVE = "https://archive.d-robotics.cc/downloads/rdk_model_zoo"

# The reviewed publication: two assets per target, exact references and URLs.
EXPECTED = {
    "x5": (
        ("x5:fasterrcnn:frcnn_mbv3_320_parta.bin",
         f"{ARCHIVE}/rdk_x5/fasterrcnn/frcnn_mbv3_320_parta.bin"),
        ("x5:fasterrcnn:frcnn_mbv3_320_partb.bin",
         f"{ARCHIVE}/rdk_x5/fasterrcnn/frcnn_mbv3_320_partb.bin")),
    "s100": (
        ("s:fasterrcnn:s100/frcnn_mbv3_320_parta.hbm",
         f"{ARCHIVE}/rdk_s100/fasterrcnn/s100/frcnn_mbv3_320_parta.hbm"),
        ("s:fasterrcnn:s100/frcnn_mbv3_320_partb.hbm",
         f"{ARCHIVE}/rdk_s100/fasterrcnn/s100/frcnn_mbv3_320_partb.hbm")),
    "s100p": (
        ("s:fasterrcnn:s100p/frcnn_mbv3_320_parta.hbm",
         f"{ARCHIVE}/rdk_s100/fasterrcnn/s100p/frcnn_mbv3_320_parta.hbm"),
        ("s:fasterrcnn:s100p/frcnn_mbv3_320_partb.hbm",
         f"{ARCHIVE}/rdk_s100/fasterrcnn/s100p/frcnn_mbv3_320_partb.hbm")),
    "s600": (
        ("s:fasterrcnn:s600/frcnn_mbv3_320_parta.hbm",
         f"{ARCHIVE}/rdk_s600/fasterrcnn/s600/frcnn_mbv3_320_parta.hbm"),
        ("s:fasterrcnn:s600/frcnn_mbv3_320_partb.hbm",
         f"{ARCHIVE}/rdk_s600/fasterrcnn/s600/frcnn_mbv3_320_partb.hbm")),
}


class PairResolutionTests(unittest.TestCase):
    """Manifest rows, URLs and descriptor digests for every target."""

    def setUp(self):
        self.pairs = DOWNLOAD.descriptor_pairs()

    def test_every_target_resolves_the_published_reference_and_url(self):
        for target, rows in EXPECTED.items():
            for (reference, url), asset in zip(rows, DOWNLOAD.pair_assets(target, self.pairs)[1]):
                with self.subTest(target=target, reference=reference):
                    self.assertEqual(asset.reference, reference)
                    self.assertEqual(asset.url, url)
                    pair = self.pairs[target]
                    expected_digest = (pair.parta_sha256 if "parta" in reference
                                       else pair.partb_sha256)
                    self.assertEqual(asset.sha256, expected_digest)

    def test_manifest_digest_mismatch_against_descriptor_is_refused(self):
        pair = self.pairs["x5"]
        drifted = SimpleNamespace(
            parta_file=pair.parta_file, partb_file=pair.partb_file,
            parta_sha256="0" * 64, partb_sha256=pair.partb_sha256)
        with self.assertRaisesRegex(ValueError, "does not match the descriptor"):
            DOWNLOAD.pair_assets("x5", {"x5": drifted})

    def test_missing_manifest_row_is_refused(self):
        real = list(DOWNLOAD.pair_assets("x5", self.pairs)[1])
        with mock.patch.object(DOWNLOAD, "list_assets",
                               return_value=real[:1]):
            with self.assertRaisesRegex(ValueError, "no fasterrcnn asset"):
                DOWNLOAD.pair_assets("x5", self.pairs)


class DestinationLayoutTests(unittest.TestCase):
    """Downloads land on the runtime's default lookup paths."""

    def setUp(self):
        self.pairs = DOWNLOAD.descriptor_pairs()

    def _capture(self, output_dir):
        captured = []

        def fake_download(asset, destination):
            captured.append((asset.reference, Path(destination)))
            return asset.sha256

        stdout = io.StringIO()
        with mock.patch.object(DOWNLOAD, "download_asset", fake_download), \
                contextlib.redirect_stdout(stdout):
            paths = DOWNLOAD.download_target("s100p", output_dir, self.pairs)
        return captured, paths

    def test_default_output_dir_is_the_runtime_model_dir(self):
        captured, _ = self._capture(DOWNLOAD.MODEL_DIR)
        pair = self.pairs["s100p"]
        self.assertEqual([path for _, path in captured],
                         [pair.parta_path, pair.partb_path])

    def test_custom_output_dir_keeps_the_target_subdirectory(self):
        with tempfile.TemporaryDirectory() as tmp:
            captured, paths = self._capture(Path(tmp))
        self.assertEqual([path for _, path in captured], paths)
        self.assertEqual(
            [path.relative_to(tmp) for path in paths],
            [Path("s100p") / self.pairs["s100p"].parta_file,
             Path("s100p") / self.pairs["s100p"].partb_file])

    def test_rerun_targets_the_same_destinations(self):
        with tempfile.TemporaryDirectory() as tmp:
            first, _ = self._capture(Path(tmp))
            second, _ = self._capture(Path(tmp))
        self.assertEqual(first, second)


class MainBehaviorTests(unittest.TestCase):
    """Argument handling and the exit-2 failure path."""

    def test_default_invocation_covers_all_four_targets(self):
        calls = []

        def fake_download(asset, destination):
            calls.append(asset.reference)
            return asset.sha256

        with mock.patch.object(DOWNLOAD, "download_asset", fake_download):
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                code = DOWNLOAD.main([])
        self.assertEqual(code, 0)
        self.assertEqual(calls, [reference
                                 for rows in EXPECTED.values()
                                 for reference, _ in rows])

    def test_transport_failure_exits_two_without_stderr_silence(self):
        def failing_download(asset, destination):
            raise OSError("connection reset")

        stderr = io.StringIO()
        with mock.patch.object(DOWNLOAD, "download_asset", failing_download), \
                contextlib.redirect_stderr(stderr):
            code = DOWNLOAD.main(["--target", "x5"])
        self.assertEqual(code, 2)
        self.assertIn("error: connection reset", stderr.getvalue())


class VerifiedInstallTests(unittest.TestCase):
    """The real shared downloader, fed by a stubbed HTTP source."""

    PAYLOAD = bytes(range(256)) * 512  # 128 KiB, several read chunks

    @staticmethod
    def _serve(payload, length=None):
        class Response:
            headers = {"Content-Length": str(
                len(payload) if length is None else length)}

            def __init__(self, data):
                import io as _io
                self._stream = _io.BytesIO(data)

            def read(self, block=-1):
                return self._stream.read(block)

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        return Response(payload)

    def _asset(self, digest):
        from utils.py_utils.assets import Asset
        pair = DOWNLOAD.descriptor_pairs()["x5"]
        return Asset(
            group="x5", sample_id="fasterrcnn",
            filename=pair.parta_file, format="bin",
            url=f"{ARCHIVE}/rdk_x5/fasterrcnn/{pair.parta_file}",
            sha256=digest)

    def test_verified_bytes_are_installed(self):
        import hashlib
        digest = hashlib.sha256(self.PAYLOAD).hexdigest()
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / "x5" / "parta.bin"
            with mock.patch("urllib.request.urlopen",
                            return_value=self._serve(self.PAYLOAD)):
                observed = DOWNLOAD.download_asset(self._asset(digest), destination)
            self.assertEqual(observed, digest)
            self.assertEqual(destination.read_bytes(), self.PAYLOAD)

    def test_mismatched_bytes_leave_no_file_behind(self):
        digest = "1" * 64  # Not the digest of PAYLOAD.
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / "x5" / "parta.bin"
            with mock.patch("urllib.request.urlopen",
                            return_value=self._serve(self.PAYLOAD)):
                with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                    DOWNLOAD.download_asset(self._asset(digest), destination)
            self.assertFalse(destination.exists())
            self.assertEqual(list(Path(tmp).glob("**/*.part")), [])


if __name__ == "__main__":
    unittest.main()
