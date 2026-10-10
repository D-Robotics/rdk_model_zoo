"""Native launch/report host tests; no SDK or model execution is asserted."""

import copy
import io
import json
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from samples.speech.paraformer.runtime.cpp import launcher
from samples.speech.paraformer.runtime.cpp.native_report import validate_report
from samples.speech.paraformer.runtime.python.model_binding import (
    INPUTS,
    OUTPUTS,
    VOCABULARY_DIGEST,
    resolve_selections,
)


def valid_report():
    selections = resolve_selections("s100")
    digests = ["a" * 64, "b" * 64, "c" * 64]
    manifest = Path("/tmp/prepared/manifest.json")
    entries = [
        {
            "utt_id": "one",
            "feature_file": "features.npy",
            "feature_sha256": "d" * 64,
            "feat_length": 71,
            "original_frames": 71,
            "truncated": False,
            "text": "reference",
        }
    ]
    vocabulary = [f"token{i}" for i in range(8404)]
    vocabulary[3] = "and@@"
    report = {
        "schema": "rdk-model-zoo/paraformer-native-run/v1",
        "status": "completed",
        "execution_backend": "native-sdk",
        "target": "s100",
        "manifest_sha256": "e" * 64,
        "manifest_path": str(manifest),
        "vocabulary_sha256": VOCABULARY_DIGEST,
        "inference_attempted": True,
        "inference_executed": True,
        "models": [],
        "records": [],
    }
    for selection, digest in zip(selections, digests):
        meta = {"model_name": "unit-test-metadata"}
        for side, contracts in (
            ("inputs", INPUTS[selection.stage]),
            ("outputs", OUTPUTS[selection.stage]),
        ):
            meta[side] = []
            for role, (names, shape, dtype) in contracts.items():
                strides = []
                span = 4
                for dimension in reversed(shape):
                    strides.insert(0, span)
                    span *= dimension
                meta[side].append(
                    {
                        "role": role,
                        "name": names[0],
                        "shape": list(shape),
                        "dtype": dtype,
                        "strides": strides,
                        "allocation_bytes": span,
                    }
                )
        report["models"].append(
            {
                "stage": selection.stage,
                "asset_id": selection.asset.reference,
                "path": str(selection.model_path.resolve()),
                "sha256": digest,
                "metadata": meta,
            }
        )
    report["records"] = [
        {
            "utt_id": "one",
            "feature_path": "/tmp/prepared/features.npy",
            "feature_sha256": "d" * 64,
            "valid_frames": 71,
            "original_frames": 71,
            "truncated": False,
            "source_record": entries[0],
            "reference_text": "reference",
            "text": "andand",
            "token_ids": [3, 3],
            "token_count": 2,
            "decoder_executed": True,
            "timings": {
                "encoder_ms": 1,
                "predictor_ms": 1,
                "cif_ms": 0.1,
                "decoder_ms": 1,
            },
        }
    ]
    arguments = (selections, digests, entries, manifest, "e" * 64, vocabulary)
    return report, arguments


class NativeCLI(unittest.TestCase):
    def test_complete_report_and_zero_token_report(self):
        report, args = valid_report()
        validate_report(report, *args)
        record = report["records"][0]
        record.update(text="", token_ids=[], token_count=0, decoder_executed=False)
        record["timings"]["decoder_ms"] = None
        validate_report(report, *args)

    def test_report_cannot_promote_host_or_malformed_results(self):
        report, args = valid_report()
        mutations = [
            lambda r: r.update(execution_backend="host-fixture"),
            lambda r: r.update(status="failed"),
            lambda r: r.update(inference_executed=None),
            lambda r: r["models"].pop(),
            lambda r: r["models"][0].update(sha256="f" * 64),
            lambda r: r["models"][0]["metadata"]["inputs"][0].update(strides=[4, 4, 4]),
            lambda r: r["models"][0]["metadata"]["inputs"][0].update(dtype="int32"),
            lambda r: r["models"][0]["metadata"]["inputs"][0].update(role="wrong"),
            lambda r: r["records"][0].update(text="invented"),
            lambda r: r["records"][0].update(valid_frames=True),
            lambda r: r["records"][0].update(token_ids=[True, 3]),
            lambda r: r["records"][0].update(feature_sha256="0" * 64),
            lambda r: r["records"][0]["timings"].update(cif_ms=float("nan")),
            lambda r: r["records"][0].update(decoder_executed=False),
        ]
        for i, mutate in enumerate(mutations):
            with self.subTest(case=i):
                bad = copy.deepcopy(report)
                mutate(bad)
                with self.assertRaises((ValueError, TypeError)):
                    validate_report(bad, *args)

    def test_report_rejects_consistently_invalid_frame_contract(self):
        report, args = valid_report()
        args[2][0]["feat_length"] = 0
        report["records"][0]["valid_frames"] = 0
        with self.assertRaises(ValueError):
            validate_report(report, *args)

    def test_host_preview_and_invalid_groups(self):
        for argv, expected in [
            (["--list-models"], 0),
            (["--target", "s100", "--dry-run"], 0),
            (["--dry-run"], 2),
            (["--target", "s100p", "--dry-run"], 2),
            (["--target", "s100", "--dry-run", "--encoder-model-path", "other.hbm"], 2),
            (["--target", "s100", "--dry-run", "--max-utts", "-1"], 2),
        ]:
            with self.subTest(argv=argv), redirect_stdout(
                io.StringIO()
            ), redirect_stderr(io.StringIO()):
                self.assertEqual(launcher.main(argv), expected)

    def test_local_gate_before_output(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(
            launcher,
            "require_execution_target",
            side_effect=ValueError("no local board"),
        ), redirect_stderr(io.StringIO()):
            output = Path(tmp) / "result"
            self.assertEqual(
                launcher.main(["--target", "s100", "--output-dir", str(output)]), 2
            )
            self.assertFalse(output.exists())

    def test_fixture_binary_is_rejected_before_inference(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(
            launcher, "require_execution_target", return_value="s100"
        ), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            tmp = Path(tmp)
            output = tmp / "out"
            manifest = tmp / "prepared.json"
            manifest.write_text("[{}]")
            vocabulary = (
                launcher.ROOT
                / "samples/speech/paraformer/tests/fixtures/published-tokens.json"
            )
            binary = tmp / "fixture"
            binary.write_text('#!/bin/sh\nprintf "Backend: host-fixture\\n"\n')
            binary.chmod(0o755)
            argv = [
                "--target",
                "s100",
                "--binary",
                str(binary),
                "--manifest",
                str(manifest),
                "--vocab-file",
                str(vocabulary),
                "--output-dir",
                str(output),
            ]
            for selection in resolve_selections("s100"):
                model = tmp / (selection.stage + ".hbm")
                model.write_text("explicit fixture")
                argv += [
                    f"--{selection.stage}-model-path",
                    str(model),
                    f"--{selection.stage}-asset-id",
                    selection.asset.reference,
                ]
            self.assertEqual(launcher.main(argv), 2)
            report = json.loads((output / "launch-report.json").read_text())
            self.assertEqual(report["status"], "failed")
            self.assertFalse(report["native_process_started"])
            self.assertEqual(len(report["processes"]), 1)
            self.assertFalse((output / "result").exists())


if __name__ == "__main__":
    unittest.main()
