"""Host export entry validation; no Torch import or model execution required."""

import argparse
import contextlib
import io
from pathlib import Path
import shutil
import tempfile
import unittest

from samples.speech.paraformer.conversion import export

ROOT = Path(__file__).resolve().parents[4]


class ExportCLI(unittest.TestCase):
    def test_help_lists_all_explicit_inputs(self):
        text = export.build_parser().format_help()
        for name in ("--model-dir", "--output-dir", "--feature", "--threads"):
            self.assertIn(name, text)

    def test_preflight_binds_pinned_support_files_and_refuses_output_reuse(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source"
            source.mkdir()
            (source / "model.pt").write_bytes(b"dummy: preflight only, never loaded")
            archived = ROOT / "platforms/s/samples/speech/paraformer/model"
            shutil.copyfile(archived / "am.mvn", source / "am.mvn")
            shutil.copyfile(archived / "paraformer_config.yaml", source / "config.yaml")
            shutil.copyfile(
                ROOT
                / "docs/releases/unified-migration/evidence/2026-09-28-b10-paraformer-pipeline/published-tokens.json",
                source / "tokens.json",
            )
            args = argparse.Namespace(
                model_dir=source,
                output_dir=Path(temporary) / "out",
                feature=[],
                threads=1,
            )
            files, identities = export.preflight(args)
            self.assertEqual(
                set(files), {"model.pt", "config.yaml", "tokens.json", "am.mvn"}
            )
            self.assertEqual(len(identities["model.pt"]["sha256"]), 64)
            self.assertFalse(args.output_dir.exists())
            (source / "tokens.json").write_text("[]")
            with self.assertRaisesRegex(ValueError, "pinned Paraformer tokens.json"):
                export.preflight(args)
            args.output_dir.mkdir()
            with self.assertRaisesRegex(ValueError, "must be new"):
                export.preflight(args)

    def test_missing_inputs_and_bad_threads_fail_without_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "out"
            for extra in ([], ["--threads", "0"]):
                with contextlib.redirect_stderr(io.StringIO()):
                    self.assertEqual(
                        export.main(
                            [
                                "--model-dir",
                                temporary,
                                "--output-dir",
                                str(output),
                                *extra,
                            ]
                        ),
                        2,
                    )
                self.assertFalse(output.exists())

    def test_comparison_rejects_shape_dtype_nonfinite_and_numeric_errors(self):
        import numpy as np

        expected = np.array([1, 2], np.float32)
        for actual in (
            expected.reshape(1, 2),
            expected.astype(np.float64),
            np.array([np.nan, 2], np.float32),
            expected + 1,
        ):
            with self.assertRaises((ValueError, AssertionError)):
                export.compare([expected], [actual])
        self.assertEqual(export.compare([expected], [expected.copy()]), [0.0])


if __name__ == "__main__":
    unittest.main()
