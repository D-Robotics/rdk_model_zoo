"""Evaluation provenance/failure boundaries; fixtures are not HMCT certification."""

import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from samples.speech.paraformer.evaluator import main, backends
from samples.speech.paraformer.runtime.python.model_binding import (
    INPUTS,
    OUTPUTS,
    bind_stage_io,
)


def metadata(stage, alias=False, count=False):
    value = {"model_name": stage}
    for side, contracts in (("input", INPUTS[stage]), ("output", OUTPUTS[stage])):
        rows = [
            (names[-1] if alias else names[0], shape, dtype)
            for names, shape, dtype in contracts.values()
        ]
        if side == "output" and count:
            rows.append(("token_num", (1,), "int32"))
        rows.reverse()
        value[f"{side}_names"] = [r[0] for r in rows]
        value[f"{side}_shapes"] = {r[0]: r[1] for r in rows}
        value[f"{side}_dtypes"] = {r[0]: r[2] for r in rows}
    return value


class FakeStage:
    def __init__(self, path, stage, pipeline, threads):
        self.stage = stage
        self.metadata = metadata(stage)
        self.inputs, self.outputs = bind_stage_io(stage, self.metadata)

    def forward(self, feed):
        # Zero fires exercises the real pipeline's decoder bypass and empty hypothesis.
        if self.stage == "decoder":
            raise AssertionError("zero-token decoder must not execute")
        return {
            names[0]: np.zeros(shape, dtype)
            for names, shape, dtype in OUTPUTS[self.stage].values()
        }


class Evaluation(unittest.TestCase):
    def args(self, root):
        vocab = root / "tokens.json"
        vocab.write_text(json.dumps([f"t{i}" for i in range(8404)]))
        feature = root / "a.npy"
        np.save(feature, np.zeros((1, 400, 560), np.float32))
        manifest = root / "manifest.json"
        manifest.write_text(
            json.dumps(
                [
                    {
                        "utt_id": "a",
                        "text": "你好",
                        "feat_length": 10,
                        "feature_file": "a.npy",
                    }
                ]
            )
        )
        models = {}
        for stage in INPUTS:
            models[stage] = root / f"{stage}.onnx"
            models[stage].write_bytes(b"explicit test fixture; not ONNX")
        return argparse.Namespace(
            pipeline="fp32",
            threads=1,
            max_utts=0,
            manifest=manifest,
            vocab=vocab,
            output_dir=root / "out",
            **models,
        )

    def run_fixture(self, args):
        digest = hashlib.sha256(args.vocab.read_bytes()).hexdigest()
        with patch.object(main, "VOCABULARY_DIGEST", digest), patch.object(
            main, "Stage", FakeStage
        ):
            return main.execute(args)

    def test_micro_cer_and_zero_token_decoder_bypass_are_recorded(self):
        with tempfile.TemporaryDirectory() as temp:
            args = self.args(Path(temp))
            report = self.run_fixture(args)
            self.assertEqual(report["status"], "completed")
            self.assertEqual(report["metrics"]["cer"], 1)
            self.assertEqual(report["metrics"]["errors"]["deletions"], 2)
            self.assertFalse(report["utterances"][0]["prediction"]["decoder_executed"])
            self.assertEqual(
                report["utterances"][0]["feature_sha256"],
                hashlib.sha256((Path(temp) / "a.npy").read_bytes()).hexdigest(),
            )
            with self.assertRaisesRegex(ValueError, "new"):
                self.run_fixture(args)

    def test_partial_failure_keeps_results_but_no_aggregate_acceptance(self):
        with tempfile.TemporaryDirectory() as temp:
            args = self.args(Path(temp))
            records = json.loads(args.manifest.read_text())
            records.append(
                {**records[0], "utt_id": "missing", "feature_file": "missing.npy"}
            )
            args.manifest.write_text(json.dumps(records))
            with self.assertRaises(FileNotFoundError):
                self.run_fixture(args)
            report = json.loads((args.output_dir / "evaluation.json").read_text())
            self.assertEqual(report["status"], "failed")
            self.assertEqual(report["current_utterance"], "missing")
            self.assertEqual(len(report["utterances"]), 1)
            self.assertIsNone(report["metrics"])

    def test_undefined_cer_for_empty_references(self):
        with tempfile.TemporaryDirectory() as temp:
            args = self.args(Path(temp))
            records = json.loads(args.manifest.read_text())
            records[0]["text"] = ""
            args.manifest.write_text(json.dumps(records))
            self.assertIsNone(self.run_fixture(args)["metrics"]["cer"])

    def test_binding_accepts_alias_reordered_io_optional_count(self):
        inputs, outputs = bind_stage_io(
            "decoder", metadata("decoder", alias=True, count=True)
        )
        self.assertEqual(inputs["acoustic"], "shape_8609")
        self.assertEqual(outputs["count"], "token_num")
        bad = metadata("decoder", count=True)
        bad["output_dtypes"]["token_num"] = "float32"
        with self.assertRaises(ValueError):
            bind_stage_io("decoder", bad)

    def test_hmct_forward_delegation_and_output_validation(self):
        # Only the adapter protocol is tested here, with no real HMCT dependency.
        from types import SimpleNamespace
        from unittest.mock import Mock

        session = Mock()
        session.forward.return_value = {
            "/encoder/after_norm/Add_1_output_0": np.zeros((1, 400, 512), np.float32)
        }
        constructor = Mock(return_value=SimpleNamespace(create_session=lambda: session))
        with patch.dict(
            "sys.modules",
            {
                "hmct": SimpleNamespace(),
                "hmct.executor": SimpleNamespace(ORTExecutor=constructor),
            },
        ), patch.object(backends, "graph_metadata", return_value=metadata("encoder")):
            stage = backends.Stage("fixture.onnx", "encoder", "int16")
            feed = {"speech": np.zeros((1, 400, 560), np.float32)}
            result = stage.forward(feed)
            constructor.assert_called_once_with("fixture.onnx")
            self.assertTrue(next(iter(result.values())).flags.owndata)
            session.forward.return_value = {"wrong": np.zeros(1, np.float32)}
            with self.assertRaises(ValueError):
                stage.forward(feed)


if __name__ == "__main__":
    unittest.main()
