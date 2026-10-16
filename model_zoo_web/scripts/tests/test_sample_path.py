import importlib.util
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock


SCRIPT = Path(__file__).resolve().parents[1] / "build_catalog.py"
SPEC = importlib.util.spec_from_file_location("build_catalog", SCRIPT)
build_catalog = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(build_catalog)

LABEL = "fixture.sample_path"
ENV = build_catalog.SAMPLE_REF_ENV


def git(root, *args):
    subprocess.run(["git", "-C", str(root), "-c", "user.name=t", "-c", "user.email=t@example.invalid", *args],
                   check=True, capture_output=True)


class SamplePathTest(unittest.TestCase):
    """The Web branch may omit a sample's source if the sample exists on the reference branch."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        git(self.root, "init", "-q", "-b", "develop")
        sample = self.root / "samples/vision/demo"
        sample.mkdir(parents=True)
        (sample / "README.md").write_text("demo\n")
        (self.root / "samples/vision/file_not_dir").write_text("x\n")
        git(self.root, "add", "-A")
        git(self.root, "commit", "-q", "-m", "develop sample")
        git(self.root, "checkout", "-q", "-b", "model_zoo_web")
        git(self.root, "rm", "-rq", "samples/vision/demo")  # the Web branch carries no sample source
        git(self.root, "commit", "-q", "-m", "web branch without sample source")
        self.patch_root = mock.patch.object(build_catalog, "ROOT", self.root)
        self.patch_root.start()
        self.addCleanup(self.patch_root.stop)
        self.env = mock.patch.dict(os.environ, {}, clear=False)
        self.env.start()
        self.addCleanup(self.env.stop)
        os.environ.pop(ENV, None)

    def test_local_directory_needs_no_reference(self):
        (self.root / "samples/vision/here").mkdir()
        self.assertEqual(build_catalog.validate_sample_path("samples/vision/here", LABEL), "samples/vision/here")

    def test_missing_locally_and_no_reference_fails_and_names_the_variable(self):
        with self.assertRaisesRegex(build_catalog.CatalogError, ENV):
            build_catalog.validate_sample_path("samples/vision/demo", LABEL)

    def test_missing_locally_but_present_on_reference_passes(self):
        os.environ[ENV] = "develop"
        self.assertEqual(build_catalog.validate_sample_path("samples/vision/demo", LABEL), "samples/vision/demo")

    def test_missing_everywhere_fails(self):
        os.environ[ENV] = "develop"
        with self.assertRaisesRegex(build_catalog.CatalogError, "locally or in develop"):
            build_catalog.validate_sample_path("samples/vision/nothing", LABEL)

    def test_a_file_on_the_reference_is_not_a_sample_directory(self):
        os.environ[ENV] = "develop"
        with self.assertRaisesRegex(build_catalog.CatalogError, "locally or in develop"):
            build_catalog.validate_sample_path("samples/vision/file_not_dir", LABEL)

    def test_unknown_reference_fails(self):
        os.environ[ENV] = "no-such-branch"
        with self.assertRaisesRegex(build_catalog.CatalogError, "locally or in no-such-branch"):
            build_catalog.validate_sample_path("samples/vision/demo", LABEL)

    def test_reference_must_be_a_plain_ref(self):
        for value in ("--exec=x", "a b", "develop;ls", "-develop"):
            with self.subTest(value=value):
                os.environ[ENV] = value
                with self.assertRaisesRegex(build_catalog.CatalogError, "plain git ref"):
                    build_catalog.validate_sample_path("samples/vision/demo", LABEL)

    def test_path_rules_are_unchanged(self):
        os.environ[ENV] = "develop"
        for bad in ("/abs/samples/x", "samples/../etc", "docs/x"):
            with self.subTest(bad=bad):
                with self.assertRaisesRegex(build_catalog.CatalogError, "safe path below samples"):
                    build_catalog.validate_sample_path(bad, LABEL)


if __name__ == "__main__":
    unittest.main()
