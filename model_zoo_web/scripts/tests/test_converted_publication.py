"""Publication refuses stale identities and preserves immutable model bytes."""
import contextlib
import errno
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stage_converted import stage
from upload_archive_model import INSTALL_MODEL
from upload_archive_assets import REMOTE_INSTALL, validate_package


class ConvertedPublicationTests(unittest.TestCase):
    def test_metadata_stays_private_until_complete_and_rejects_replacement(self):
        namespace = {"__name__": "installer_test"}
        exec(REMOTE_INSTALL, namespace)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            public = root / "public"
            public.mkdir()
            private_home = root / "other-filesystem-home"
            private_home.mkdir()
            prefix = "assets/rdk_s100/modnet/new-model/v1"
            destination = public / prefix
            files = {name: ("complete " + name).encode() for name in
                     ("release.json", "release.source.json", "oe_report.html", "oe_report_data.json", "SHA256SUMS")}
            original_rename = os.rename
            def publish(source, target):
                source = Path(source)
                self.assertEqual(source.parent, destination.parent.resolve())
                self.assertEqual(source.stat().st_mode & 0o777, 0o700)
                self.assertEqual({p.name: p.read_bytes() for p in source.iterdir()}, files)
                self.assertFalse(destination.exists())
                return original_rename(source, target)
            with patch.object(Path, "home", return_value=private_home), patch.object(os, "rename", side_effect=publish):
                self.assertEqual(namespace["install"](public, prefix, files)["status"], "created")
            self.assertEqual(destination.stat().st_mode & 0o777, 0o755)
            self.assertEqual(namespace["install"](public, prefix, files)["status"], "identical-existing")
            with self.assertRaises(ValueError):
                namespace["install"](public, prefix, dict(files, **{"release.json": b"replacement"}))
            self.assertEqual((destination / "release.json").read_bytes(), files["release.json"])

    def test_initial_staging_preserves_sources_and_rejects_stale_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / 'new.bin'
            model.write_bytes(b'real compiled test fixture')
            digest = hashlib.sha256(model.read_bytes()).hexdigest()
            manifest = {'status': 'converted', 'metadata_revision': 1,
                'model': {'source': 'convnext'}, 'target': {'platform': 'x5'},
                'artifact': {'filename': model.name, 'sha256': digest,
                    'size_bytes': model.stat().st_size,
                    'url': 'https://archive.d-robotics.cc/downloads/rdk_model_zoo_web/rdk_x5/convnext/new.bin'},
                'validation': {'artifact_sha256': digest, 'conversion': 'passed', 'host_smoke': 'passed',
                    **{k: 'not-run' for k in ('board_smoke', 'accuracy', 'runtime_1', 'runtime_2', 'cpp_1', 'cpp_2')}}}
            report = root / 'report.html'
            report.write_text('<!doctype html><html>original compiler report</html>')
            oe = root / 'oe.json'
            oe.write_text(json.dumps({'provenance': {'artifact_sha256': digest}}))
            output = root / 'package'
            stage(manifest, model, report, oe, output)
            self.assertEqual(json.loads((output / 'release.source.json').read_text()), manifest)
            self.assertEqual((output / 'oe_report.html').read_bytes(), report.read_bytes())
            self.assertTrue(validate_package(output)[0].endswith('/v1'))
            manifest['accuracy'] = {'top1': 71.75}
            with self.assertRaises(ValueError):
                stage(manifest, model, report, oe, root / 'stale')
            self.assertFalse((root / 'stale').exists())

    def test_model_install_preserves_other_checksums_and_rejects_replacement(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            public = root / 'public'
            public.mkdir()
            directory = public / 'rdk_x5/convnext'
            directory.mkdir(parents=True)
            (directory / 'SHA256SUMS').write_text('a' * 64 + '  historical.bin\n')

            def install(data):
                staged = root / 'private-upload'
                staged.write_bytes(data)
                digest = hashlib.sha256(data).hexdigest()
                args = ['installer', str(public), 'rdk_x5/convnext', 'new.bin', digest, str(len(data)), str(staged)]
                original_link = os.link
                def cross_device_link(source, destination):
                    if Path(source).parent == root:
                        raise OSError(errno.EXDEV, 'Upload home is another filesystem')
                    self.assertEqual(Path(source).parent.stat().st_mode & 0o777, 0o700)
                    return original_link(source, destination)
                with patch.object(sys, 'argv', args), patch.object(Path, 'home', return_value=root), \
                        patch.object(os, 'link', side_effect=cross_device_link), contextlib.redirect_stdout(io.StringIO()):
                    exec(INSTALL_MODEL, {})

            install(b'accepted binary')
            install(b'accepted binary')  # Resume is harmless.
            with self.assertRaises(ValueError):
                install(b'different binary')
            self.assertEqual((directory / 'new.bin').read_bytes(), b'accepted binary')
            self.assertIn('historical.bin', (directory / 'SHA256SUMS').read_text())


if __name__ == '__main__':
    unittest.main()
