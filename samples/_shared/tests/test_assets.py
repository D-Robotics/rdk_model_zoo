"""Existing manifest identities remain distinct and are never executable input."""
import unittest
from pathlib import Path
import tempfile
import io
import hashlib
from unittest.mock import patch


class ManifestAssetsTests(unittest.TestCase):
    def test_truncated_http_response_is_not_installed_without_publisher_hash(self):
        from http.client import HTTPResponse
        from samples._shared.assets import Asset, download_asset

        class Socket:
            def makefile(self, mode):
                return io.BytesIO(b'HTTP/1.1 200 OK\r\nContent-Length: 100\r\n\r\ntruncated')

        response = HTTPResponse(Socket())
        response.begin()
        asset = Asset('x5', 'fixture', 'fixture.bin', 'bin', 'https://example.invalid/model', None)
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / 'asset.bin'
            with patch('urllib.request.urlopen', return_value=response):
                with self.assertRaisesRegex(ValueError, 'length'):
                    download_asset(asset, destination)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_qualified_reference_keeps_platform_and_sample_identity(self):
        from samples._shared.assets import resolve_asset
        item = resolve_asset('x5:resnet:resnet18_224x224_nv12.bin')
        self.assertEqual(item.sample_id, 'resnet')
        self.assertEqual(item.reference, 'x5:resnet:resnet18_224x224_nv12.bin')
        self.assertEqual(item.url, 'https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_x5/resnet18_224x224_nv12.bin')
        self.assertIsNone(item.sha256)

    def test_manifest_authority_is_the_unified_release_location(self):
        """A4: manifests live at docs/release/{group}/, not under platforms/."""
        from samples._shared import assets
        from samples._shared.assets import resolve_asset
        for group in ('x5', 's'):
            location = assets._ROOT / f'docs/release/{group}/models.yaml'
            self.assertTrue(location.is_file(), f'missing unified manifest {location}')
            self.assertIn(f'docs/release/{group}/models.yaml', str(location))
        item = resolve_asset('s:resnet18:s100/resnet18_224x224_nv12.hbm')
        self.assertEqual(item.source_path, 'docs/release/s/models.yaml')

    def test_group_and_paths_cannot_escape_manifest_scope(self):
        from samples._shared.assets import resolve_asset
        for reference in ('../s:resnet18:x', 's:resnet18:../../x', 'resnet18', 's:missing:missing'):
            with self.subTest(reference=reference), self.assertRaises(ValueError):
                resolve_asset(reference)

    def test_s_classification_does_not_invent_s100p_artifact(self):
        from samples._shared.assets import list_assets
        records = list_assets('s', 'resnet18')
        self.assertEqual({a.filename for a in records}, {'s100/resnet18_224x224_nv12.hbm', 's600/resnet18_224x224_nv12.hbm'})

    def test_download_validates_hash_before_installing_file(self):
        from samples._shared.assets import Asset, download_asset
        asset = Asset('x5', 'fixture', 'fixture.bin', 'bin', 'https://example.invalid/model', hashlib.sha256(b'good').hexdigest())
        with tempfile.TemporaryDirectory(prefix='model with spaces ') as directory:
            destination = Path(directory) / 'asset.bin'
            with patch('urllib.request.urlopen', return_value=io.BytesIO(b'bad')):
                with self.assertRaises(ValueError):
                    download_asset(asset, destination)
            self.assertFalse(destination.exists())
            self.assertEqual(list(Path(directory).iterdir()), [])
            with patch('urllib.request.urlopen', return_value=io.BytesIO(b'good')):
                download_asset(asset, destination)
            self.assertEqual(destination.read_bytes(), b'good')

    def test_empty_download_is_not_a_complete_model(self):
        from samples._shared.assets import Asset, download_asset
        asset = Asset('s', 'fixture', 'fixture.hbm', 'hbm', 'https://example.invalid/model', None)
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / 'asset.hbm'
            with patch('urllib.request.urlopen', return_value=io.BytesIO()):
                with self.assertRaises(ValueError):
                    download_asset(asset, destination)
            self.assertFalse(destination.exists())


if __name__ == '__main__':
    unittest.main()
