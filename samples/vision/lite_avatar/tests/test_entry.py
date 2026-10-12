"""Entry-point behavior for the model-free and failing modes of main()."""

import io
import sys
import unittest
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))

from samples.vision.lite_avatar.runtime.python import main as main_module


class MainModelFreeModes(unittest.TestCase):
    def test_list_models_lists_every_published_set(self):
        buffer, errbuf = io.StringIO(), io.StringIO()
        with redirect_stdout(buffer), redirect_stderr(errbuf):
            code = main_module.main(["--list-models"])
        self.assertEqual(code, 0)
        text = buffer.getvalue()
        for target in ("s100", "s100p", "s600"):
            self.assertIn(f"s:lite_avatar:{target}/", text)
            self.assertIn(f"[{target}]", text)
        self.assertEqual(errbuf.getvalue(), "")

    def test_dry_run_resolves_and_reports_missing_artifacts(self):
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            code = main_module.main(["--target", "s100", "--dry-run"])
        self.assertEqual(code, 1)
        self.assertIn("stage: audio_encoder", buffer.getvalue())
        self.assertIn("Missing artifacts", buffer.getvalue())

    def test_real_run_without_prepared_models_fails_with_pointers(self):
        wav = (
            Path(main_module.__file__).resolve().parents[2]
            / "test_data"
            / "asr_example.wav"
        )
        buffer, errbuf = io.StringIO(), io.StringIO()
        with redirect_stdout(buffer), redirect_stderr(errbuf):
            code = main_module.main(["--target", "s100", "--audio", str(wav)])
        self.assertEqual(code, 2)
        self.assertIn("Missing model artifacts", errbuf.getvalue())
        self.assertIn("model/README.md", errbuf.getvalue())

    def test_missing_audio_file_is_reported_before_model_checks(self):
        errbuf = io.StringIO()
        with redirect_stdout(io.StringIO()), redirect_stderr(errbuf):
            code = main_module.main(["--target", "s100", "--audio", "whatever.wav"])
        self.assertEqual(code, 2)
        self.assertIn("Missing audio file", errbuf.getvalue())

    def test_unpublished_targets_and_bad_flags_fail_cleanly(self):
        errbuf = io.StringIO()
        with redirect_stdout(io.StringIO()), redirect_stderr(errbuf):
            code = main_module.main(["--target", "x5", "--audio", "x.wav"])
        self.assertEqual(code, 2)
        self.assertIn("error:", errbuf.getvalue())

        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            code = main_module.main(["--target", "auto", "--dry-run"])
        self.assertEqual(code, 2)

        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            code = main_module.main(
                ["--target", "s100", "--audio", "x.wav", "--priority", "300"]
            )
        self.assertEqual(code, 2)

        with self.assertRaises(SystemExit):
            main_module.main(["--definitely-not-a-flag"])


if __name__ == "__main__":
    unittest.main()
