"""Download-target resolution tests for the MobileNetV2 sample.

Regression coverage for the board-found defect (X5 smoke, 2026-09-21) where
``download_target`` resolved its manifest reference without forwarding the
variant, so ``--variant medium`` fetched the small artifact.  These tests pin
the exact published reference every supported combination must resolve to,
with the network downloader stubbed out.
"""

from __future__ import annotations

from pathlib import Path
import unittest
from unittest import mock

from samples.vision.mobilenetv2.model import download as download_mod

EXPECTED_REFERENCES = {
    'x5': 'x5:mobilenetv2:mobilenetv2_224x224_nv12.bin',
    's100': 's:mobilenetv2:s100/mobilenetv2_224x224_nv12.hbm',
    's600': 's:mobilenetv2:s600/mobilenetv2_224x224_nv12.hbm',
}


class DownloadTargetReferenceTests(unittest.TestCase):
    output_dir = Path("/tmp/rdk-zoo-download-fixture")

    def test_download_target_resolves_the_published_reference_per_target(self):
        for target in EXPECTED_REFERENCES:
            expected = EXPECTED_REFERENCES[target]
            with self.subTest(target=target):
                captured = {}

                def fake_download(asset, destination):
                    captured["reference"] = asset.reference
                    captured["filename"] = asset.filename
                    captured["destination"] = destination
                    return "0" * 64

                with mock.patch.object(download_mod, "download_asset", fake_download):
                    digest = download_mod.download_target(target, self.output_dir)
                self.assertEqual(captured["reference"], expected)
                self.assertEqual(captured["filename"], expected.rsplit(":", 1)[1])
                self.assertEqual(
                    captured["destination"], self.output_dir / captured["filename"]
                )
                self.assertEqual(digest, "0" * 64)


if __name__ == "__main__":
    unittest.main()
