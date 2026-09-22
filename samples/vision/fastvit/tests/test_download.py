"""Download-target resolution tests for the FastViT sample.

The variant must be forwarded into the manifest reference.  These tests pin
the exact published reference every supported (target, variant) combination
must resolve to — both rows — with the network downloader stubbed out.
"""

from __future__ import annotations

from pathlib import Path
import unittest
from unittest import mock

from samples.vision.fastvit.model import download as download_mod

EXPECTED_REFERENCES = {
    ('x5', 's12'): 'x5:fastvit:FastViT_S12_224x224_nv12.bin',
    ('x5', 'sa12'): 'x5:fastvit:FastViT_SA12_224x224_nv12.bin',
    ('x5', 't12'): 'x5:fastvit:FastViT_T12_224x224_nv12.bin',
    ('x5', 't8'): 'x5:fastvit:FastViT_T8_224x224_nv12.bin',
}


class DownloadTargetReferenceTests(unittest.TestCase):
    output_dir = Path("/tmp/rdk-zoo-download-fixture")

    def test_download_target_resolves_the_published_reference_per_target_variant(self):
        for (target, variant), expected in EXPECTED_REFERENCES.items():
            with self.subTest(target=target, variant=variant):
                captured = {}

                def fake_download(asset, destination):
                    captured["reference"] = asset.reference
                    captured["filename"] = asset.filename
                    captured["destination"] = destination
                    return "0" * 64

                with mock.patch.object(download_mod, "download_asset", fake_download):
                    digest = download_mod.download_target(
                        target, self.output_dir, variant=variant
                    )
                self.assertEqual(captured["reference"], expected)
                self.assertEqual(captured["filename"], expected.rsplit(":", 1)[1])
                self.assertEqual(
                    captured["destination"], self.output_dir / captured["filename"]
                )
                self.assertEqual(digest, "0" * 64)

    def test_default_download_target_is_s12_on_x5(self):
        captured = {}

        def fake_download(asset, destination):
            captured["reference"] = asset.reference
            return "1" * 64

        with mock.patch.object(download_mod, "download_asset", fake_download):
            download_mod.download_target("x5", self.output_dir)
        # The source entrypoint defaulted to the S12 (lowercase id `s12`) artifact.
        self.assertEqual(
            captured["reference"], "x5:fastvit:FastViT_S12_224x224_nv12.bin"
        )

    def test_unpublished_combination_is_an_explicit_error(self):
        with self.assertRaises(ValueError):
            download_mod.download_target("s100", self.output_dir, variant="s12")


    def test_omitted_variant_delegates_parser_default_through_main(self):
        """B3-R1 regression: `download.py --target x5` (no --variant) must
        run end-to-end through parser -> main -> download_target and fetch
        the sample's default-variant reference — earlier tests called
        download_target directly and missed an illegal parser default."""

        captured = {}

        def fake_download(asset, destination):
            captured["reference"] = asset.reference
            return "2" * 64

        with mock.patch.object(download_mod, "download_asset", fake_download):
            rc = download_mod.main(["--target", "x5", "--output-dir", str(self.output_dir)])
        self.assertEqual(rc, 0)
        self.assertEqual(captured["reference"], "x5:fastvit:FastViT_S12_224x224_nv12.bin")
        # an explicitly illegal variant still fails at the parser: argparse
        # prints the usage error and raises SystemExit(2)
        with mock.patch.object(download_mod, "download_asset", fake_download), \
                mock.patch("sys.stderr"):
            with self.assertRaises(SystemExit) as ctx:
                download_mod.main(["--target", "x5", "--variant", "base",
                                   "--output-dir", str(self.output_dir)])
        self.assertEqual(ctx.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
