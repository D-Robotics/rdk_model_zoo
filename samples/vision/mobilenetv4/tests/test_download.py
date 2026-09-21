"""Download-target resolution tests for the MobileNetV4 sample.

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

from samples.vision.mobilenetv4.model import download as download_mod

EXPECTED_REFERENCES = {
    ('x5', 'small'): 'x5:mobilenetv4:MobileNetV4_conv_small_224x224_nv12.bin',
    ('x5', 'medium'): 'x5:mobilenetv4:MobileNetV4_conv_medium_224x224_nv12.bin',
    ('s100', 'small'): 's:mobilenetv4:s100/mobilenetv4_small_224x224_nv12.hbm',
    ('s600', 'small'): 's:mobilenetv4:s600/mobilenetv4_small_224x224_nv12.hbm',
    ('s100', 'medium'): 's:mobilenetv4:s100/mobilenetv4_medium_256x256_nv12.hbm',
    ('s600', 'medium'): 's:mobilenetv4:s600/mobilenetv4_medium_256x256_nv12.hbm',
}


class DownloadTargetReferenceTests(unittest.TestCase):
    output_dir = Path("/tmp/rdk-zoo-download-fixture")

    def test_download_target_resolves_the_published_reference_per_target_and_variant(self):
        for target, variant in EXPECTED_REFERENCES:
            expected = EXPECTED_REFERENCES[(target, variant)]
            with self.subTest(target=target, variant=variant):
                captured = {}

                def fake_download(asset, destination):
                    captured["reference"] = asset.reference
                    captured["filename"] = asset.filename
                    captured["destination"] = destination
                    return "0" * 64

                with mock.patch.object(download_mod, "download_asset", fake_download):
                    digest = download_mod.download_target(target, self.output_dir, variant=variant)
                self.assertEqual(captured["reference"], expected)
                self.assertEqual(captured["filename"], expected.rsplit(":", 1)[1])
                self.assertEqual(
                    captured["destination"], self.output_dir / captured["filename"]
                )
                self.assertEqual(digest, "0" * 64)


if __name__ == "__main__":
    unittest.main()
