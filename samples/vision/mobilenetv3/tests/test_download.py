"""Download-target resolution tests for the MobileNetV3 sample.

Regression coverage for the board-found defect (X5 smoke, 2026-09-21) where
``download_target`` resolved its manifest reference without forwarding the
variant, so a non-default ``--variant`` fetched the default artifact.  These tests pin
the exact published reference every supported combination must resolve to,
with the network downloader stubbed out.
"""

from __future__ import annotations

from pathlib import Path
import unittest
from unittest import mock

from samples.vision.mobilenetv3.model import download as download_mod

EXPECTED_REFERENCES = {
    ('s100', 'large'): 's:mobilenetv3:s100/mobilenetv3_large_100_nashe_224x224_nv12.hbm',
    ('s100p', 'large'): 's:mobilenetv3:s100p/mobilenetv3_large_100_nashm_224x224_nv12.hbm',
    ('s600', 'large'): 's:mobilenetv3:s600/mobilenetv3_large_100_nashp_224x224_nv12.hbm',
    ('s100', 'small'): 's:mobilenetv3:s100/mobilenetv3_small_100_nashe_224x224_nv12.hbm',
    ('s100p', 'small'): 's:mobilenetv3:s100p/mobilenetv3_small_100_nashm_224x224_nv12.hbm',
    ('s600', 'small'): 's:mobilenetv3:s600/mobilenetv3_small_100_nashp_224x224_nv12.hbm',
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
