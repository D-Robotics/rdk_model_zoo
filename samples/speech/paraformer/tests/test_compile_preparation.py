"""Recipe parity and external-process state transitions; compiler is an explicit fixture."""

import argparse
import contextlib
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
import numpy as np
import yaml

from samples._shared.assets import sha256_file
from samples.speech.paraformer.conversion.calibration import CALIBRATION
from samples.speech.paraformer.conversion.configuration import STAGES, make_config
from samples.speech.paraformer.conversion.compile import main, compile_workspace
from samples.speech.paraformer.conversion.workspace import verify_prepared

ROOT = Path(__file__).resolve().parents[4]


class CompilePreparation(unittest.TestCase):
    def make_workspace(self, root):
        (root / "source").mkdir(parents=True)
        (root / "configs").mkdir()
        shutil.copyfile(
            ROOT / "samples/speech/paraformer/model/am.mvn", root / "source/am.mvn"
        )
        (root / "source/export-report.json").write_text('{"test_fixture": true}')
        report = {
            "schema": "rdk-model-zoo/paraformer-calibration/v1",
            "status": "prepared",
            "target": "s100",
            "march": "nash-e",
            "cif_valid_frame_mask": False,
            "sample_count_selected": 1,
            "jobs": 32,
            "records": [],
            "models": {},
            "configs": {},
            "cmvn_sha256": sha256_file(root / "source/am.mvn"),
            "export_report_sha256": sha256_file(root / "source/export-report.json"),
        }
        record = {"filename": "000000.npy", "arrays": {}}
        for name, (shape, dtype) in CALIBRATION.items():
            directory = root / "calibration" / name
            directory.mkdir(parents=True)
            path = directory / "000000.npy"
            np.save(path, np.zeros(shape, dtype), allow_pickle=False)
            record["arrays"][name] = {
                "sha256": sha256_file(path),
                "shape": list(shape),
                "dtype": dtype,
            }
        report["records"].append(record)
        for stage in STAGES:
            model = root / f"source/{stage}.onnx"
            model.write_bytes(
                b"Unit-test compiler transport fixture, not an ONNX model"
            )
            report["models"][stage] = {"sha256": sha256_file(model)}
            path = root / f"configs/{stage}.yaml"
            path.write_text(yaml.safe_dump(make_config(stage), sort_keys=False))
            report["configs"][stage] = sha256_file(path)
        (root / "preparation.json").write_text(json.dumps(report))
        return report

    def compiler(self, root, mode):
        path = root / "fixture-compiler"
        # The fixture interprets no ONNX and provides no vendor-compilation evidence.
        path.write_text(
            f"#!{sys.executable}\n"
            + """import sys
from pathlib import Path
import yaml
print("EXPLICIT HOST COMPILER FIXTURE")
print("fixture diagnostic", file=sys.stderr)
config = yaml.safe_load(Path(sys.argv[2]).read_text())
"""
            + (
                """p = config['model_parameters']
out = Path(p['working_dir'])
out.mkdir()
(out/(p['output_model_file_prefix']+'.hbm')).write_bytes(b'fixture-not-hbm')
"""
                if mode == "artifact"
                else ("raise SystemExit(7)\n" if mode == "failure" else "")
            )
        )
        path.chmod(0o755)
        return path

    def test_all_nonpath_recipe_settings_match_archived_source(self):
        for stage in STAGES:
            source = yaml.safe_load(
                (
                    ROOT
                    / f"platforms/s/samples/speech/paraformer/conversion/configs/{stage}_int16.yaml"
                ).read_text()
            )
            generated = make_config(stage)
            for key in ("input_parameters", "compiler_parameters"):
                self.assertEqual(generated[key], source[key])
            self.assertEqual(
                generated["calibration_parameters"]["quant_config"],
                source["calibration_parameters"]["quant_config"],
            )
            self.assertEqual(
                generated["model_parameters"]["march"],
                source["model_parameters"]["march"],
            )
            self.assertEqual(
                generated["model_parameters"]["output_model_file_prefix"],
                source["model_parameters"]["output_model_file_prefix"],
            )

    def test_success_stays_compiled_unverified_and_preserves_both_logs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "prepared"
            self.make_workspace(root)
            output = Path(tmp) / "compile"
            args = argparse.Namespace(
                workspace=root,
                output_dir=output,
                compiler=str(self.compiler(Path(tmp), "artifact")),
            )
            report = compile_workspace(args)
            self.assertEqual(report["status"], "compiled_unverified")
            self.assertEqual(report["board"], "not-run")
            self.assertEqual(set(report["stages"]), set(STAGES))
            for stage in STAGES:
                self.assertEqual(report["stages"][stage]["returncode"], 0)
                self.assertIn("FIXTURE", (output / f"{stage}.stdout.log").read_text())
                self.assertIn(
                    "diagnostic", (output / f"{stage}.stderr.log").read_text()
                )
            verify_prepared(
                root
            )  # Compile remapping must not change preparation configs.
            with self.assertRaisesRegex(ValueError, "must be new"):
                compile_workspace(args)

    def test_nonzero_or_missing_artifact_stops_after_first_stage(self):
        for mode in ("failure", "no-artifact"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp) / "prepared"
                self.make_workspace(root)
                output = Path(tmp) / "compile"
                args = argparse.Namespace(
                    workspace=root,
                    output_dir=output,
                    compiler=str(self.compiler(Path(tmp), mode)),
                )
                with self.assertRaises(RuntimeError):
                    compile_workspace(args)
                report = json.loads((output / "compile-report.json").read_text())
                self.assertEqual(report["status"], "compile_failed")
                self.assertEqual(set(report["stages"]), {"encoder"})
                self.assertEqual(
                    report["stages"]["encoder"]["returncode"],
                    7 if mode == "failure" else 0,
                )

    def test_compiler_cannot_start_is_a_terminal_stage_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "prepared"
            self.make_workspace(root)
            compiler = Path(tmp) / "bad-executable"
            compiler.write_text("Not an executable file format")
            compiler.chmod(0o755)
            output = Path(tmp) / "compile"
            args = argparse.Namespace(
                workspace=root, output_dir=output, compiler=str(compiler)
            )
            with self.assertRaises(OSError):
                compile_workspace(args)
            report = json.loads((output / "compile-report.json").read_text())
            self.assertEqual(report["status"], "compile_failed")
            stage = report["stages"]["encoder"]
            self.assertEqual(stage["status"], "failed")
            self.assertIsNone(stage["returncode"])
            self.assertIn("finished_utc", stage)

    def test_added_or_changed_calibration_rejected_before_compile(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "prepared"
            self.make_workspace(root)
            extra = root / "calibration/speech/extra.npy"
            extra.write_bytes(b"injected")
            with self.assertRaisesRegex(ValueError, "Unexpected calibration files"):
                verify_prepared(root)
            extra.unlink()
            (root / "calibration/token_num/000000.npy").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "Calibration identity mismatch"):
                verify_prepared(root)

    def test_missing_compiler_creates_no_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "out"
            with contextlib.redirect_stderr(io.StringIO()):
                rc = main(
                    [
                        "--workspace",
                        tmp,
                        "--output-dir",
                        str(output),
                        "--compiler",
                        str(Path(tmp) / "not-an-executable"),
                    ]
                )
            self.assertEqual(rc, 2)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
