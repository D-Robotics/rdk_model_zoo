import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from samples.vision.osnet.model.download import install_bundle, install_candidate


class ArtifactTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.bundle = self.root / "delivery.zip"
        self.payload = b"synthetic model for import tests"
        self.model_hash = hashlib.sha256(self.payload).hexdigest()
        with zipfile.ZipFile(self.bundle, "w") as archive:
            archive.writestr("models/test.bin", self.payload)
            archive.writestr("SHA256SUMS", self.model_hash + "  models/test.bin\n")
            archive.writestr("../../unexpected.txt", b"must never extract")
        self.manifest = {"bundle": {"sha256": hashlib.sha256(self.bundle.read_bytes()).hexdigest()},
                         "artifacts": {"test": {"filename": "test.bin", "sha256": self.model_hash}}}
        mock = patch("samples.vision.osnet.model.download.load_manifest", return_value=self.manifest)
        mock.start()
        self.addCleanup(mock.stop)

    def test_imports_only_selected_file_and_reuses_identical_copy(self):
        output = self.root / "models"
        install_bundle(self.bundle, output, "test")
        install_bundle(self.bundle, output, "test")
        self.assertEqual((output / "test.bin").read_bytes(), self.payload)
        self.assertEqual([p.name for p in output.iterdir()], ["test.bin"])

    def test_refuses_corrupt_bundle_before_creating_output(self):
        self.bundle.write_bytes(self.bundle.read_bytes() + b"tamper")
        with self.assertRaises(ValueError):
            install_bundle(self.bundle, self.root / "models", "test")
        self.assertFalse((self.root / "models").exists())

    def test_preserves_different_existing_model(self):
        output = self.root / "models"
        output.mkdir()
        (output / "test.bin").write_bytes(b"user artifact")
        with self.assertRaises(FileExistsError):
            install_bundle(self.bundle, output, "test")
        self.assertEqual((output / "test.bin").read_bytes(), b"user artifact")

    def candidate(self, target="s100"):
        folder = self.root / "candidate" / target
        folder.mkdir(parents=True)
        (folder / "test.bin").write_bytes(self.payload)
        import json
        release = {"platform": target, "artifact": {"filename": "test.bin", "sha256": self.model_hash}}
        (folder / "release.json").write_text(json.dumps(release))
        self.manifest["artifacts"]["test"]["target"] = "s100"
        return folder

    def test_candidate_import_needs_no_private_ssh_or_fixture(self):
        folder = self.candidate()
        output = self.root / "empty-model-cache"
        install_candidate(folder.parent, output, "test")
        self.assertEqual((output / "test.bin").read_bytes(), self.payload)
        self.assertEqual([p.name for p in output.iterdir()], ["test.bin"])

    def test_candidate_rejects_wrong_platform_before_writing(self):
        folder = self.candidate("s600")
        with self.assertRaisesRegex(ValueError, "platform"):
            install_candidate(folder, self.root / "output", "test")
        self.assertFalse((self.root / "output").exists())

    def test_distribution_filename_can_differ_from_native_cache_name(self):
        import json
        folder = self.candidate()
        (folder / "test.bin").rename(folder / "published-name.bin")
        release = json.loads((folder / "release.json").read_text())
        release["artifact"]["filename"] = "published-name.bin"
        (folder / "release.json").write_text(json.dumps(release))
        output = self.root / "output"
        install_candidate(folder, output, "test")
        self.assertEqual((output / "test.bin").read_bytes(), self.payload)

    def test_candidate_rejects_changed_model_and_preserves_local_file(self):
        folder = self.candidate()
        (folder / "test.bin").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            install_candidate(folder, self.root / "output", "test")
        (folder / "test.bin").write_bytes(self.payload)
        output = self.root / "output"
        output.mkdir()
        (output / "test.bin").write_bytes(b"user model")
        with self.assertRaises(FileExistsError):
            install_candidate(folder, output, "test")
        self.assertEqual((output / "test.bin").read_bytes(), b"user model")


if __name__ == "__main__":
    unittest.main()
