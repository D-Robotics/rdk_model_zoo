"""KWS sample contracts without vendor SDK or Paddle installation."""

import contextlib
import io
import json
import unittest
from dataclasses import replace
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
from samples.speech.kws.runtime.python.model_binding import (
    resolve_selection,
    bind_model,
)
from samples.speech.kws.runtime.python.frontend import (
    Config,
    prepare_waveform,
    prepare_features,
)
from samples.speech.kws.runtime.python.kws import KWS
from samples.speech.kws.runtime.python.main import main


def metadata(dtype="float32", quant=None):
    return dict(
        model_name="kws",
        model_names=["kws"],
        input_names=["features"],
        output_names=["score"],
        input_shapes={"features": [1, 373, 80]},
        output_shapes={"score": [1, 2, 1]},
        input_dtypes={"features": "float32"},
        output_dtypes={"score": dtype},
        output_quants={"score": quant} if quant else {},
    )


class ContractTests(unittest.TestCase):
    def test_exact_asset_and_no_target_fallback(self):
        self.assertEqual(
            resolve_selection("s100").asset.reference, "s:kws:s100/kws.hbm"
        )
        for target in ("x5", "s100p", "s600"):
            with self.assertRaises(ValueError):
                resolve_selection(target)
        with self.assertRaises(ValueError):
            resolve_selection("s100", model_path="arbitrary.hbm")
        with self.assertRaises(ValueError):
            resolve_selection("s100", asset_id="s:kws:s600/kws.hbm")

    def test_forged_selection_and_wrong_metadata_rejected(self):
        selection = resolve_selection("s100")
        with self.assertRaises(ValueError):
            bind_model(replace(selection, target="s600"), metadata())
        for field, value in [
            ("input_names", ["a", "b"]),
            ("input_shapes", {"features": [1, 373, 40]}),
            ("input_dtypes", {"features": "int8"}),
            ("output_shapes", {"score": [2, 2, 1]}),
        ]:
            meta = metadata()
            meta[field] = value
            with self.assertRaises(ValueError):
                bind_model(selection, meta)

    def test_waveform_padding_truncation_is_owned(self):
        x = np.arange(10, dtype=np.float32) / 10
        y = prepare_waveform(x, 16000, Config())
        self.assertEqual(y.shape, (1, 60000))
        np.testing.assert_array_equal(y[0, :10], x)
        self.assertFalse(np.shares_memory(x, y))
        self.assertEqual(np.count_nonzero(y[0, 10:]), 0)
        long = np.zeros(60001, dtype=np.float32)
        long[-1] = 1
        self.assertEqual(np.count_nonzero(prepare_waveform(long, 16000, Config())), 0)

    def test_invalid_waveforms_and_frontend_parameters(self):
        for x, rate in [
            (np.zeros(0, np.float32), 16000),
            (np.zeros((2, 10), np.float32), 16000),
            (np.array([np.nan], np.float32), 16000),
            (np.zeros(20, np.float32), 8000),
        ]:
            with self.assertRaises(ValueError):
                prepare_waveform(x, rate, Config())
        with self.assertRaises(ValueError):
            prepare_waveform(np.ones(10, np.float32), 16000, Config(n_mels=40))

    def test_frontend_shape_and_finite_checks(self):
        binding = bind_model(resolve_selection("s100"), metadata())
        fake = lambda waveform, cfg: np.zeros((373, 80), np.float32)
        tensors = prepare_features(
            np.ones(20, np.float32), 16000, Config(), binding, fake
        )
        self.assertEqual(tensors["features"].shape, (1, 373, 80))
        for bad in (
            np.zeros((373, 40), np.float32),
            np.full((373, 80), np.nan, np.float32),
        ):
            with self.assertRaises(ValueError):
                prepare_features(
                    np.ones(20, np.float32), 16000, Config(), binding, lambda w, c: bad
                )

    def test_stages_preserve_raw_and_no_second_sigmoid(self):
        binding = bind_model(resolve_selection("s100"), metadata())
        raw = np.array([[[0.25], [0.985]]], np.float32)
        calls = []

        def runner(tensors):
            calls.append(tensors)
            return raw

        task = KWS(
            runner, binding, frontend=lambda w, c: np.zeros((373, 80), np.float32)
        )
        self.assertIs(
            task.forward({"features": np.zeros((1, 373, 80), np.float32)}), raw
        )
        self.assertAlmostEqual(
            task.predict(np.ones(10, np.float32), 16000), 0.985, places=6
        )
        self.assertEqual(len(calls), 2)
        for value in (
            np.full_like(raw, -0.1),
            np.full_like(raw, 1.1),
            np.full_like(raw, np.nan),
        ):
            with self.assertRaises(ValueError):
                task.post_process(value)

    def test_integer_output_requires_scale_and_decodes_before_max(self):
        selection = resolve_selection("s100")
        with self.assertRaises(ValueError):
            bind_model(selection, metadata("int8"))
        binding = bind_model(
            selection,
            metadata(
                "int8",
                SimpleNamespace(
                    scale=[0.01], zero_point=[0], axis=0, quant_type="SCALE"
                ),
            ),
        )
        task = KWS(lambda x: None, binding)
        self.assertAlmostEqual(
            task.post_process(np.array([[[10], [90]]], np.int8)), 0.9, places=6
        )

    def test_host_modes_do_not_load_sdk_or_frontend(self):
        for args in (["--list-models"], ["--target", "s100", "--dry-run"]):
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                self.assertEqual(main(args), 0)
            self.assertTrue(json.loads(out.getvalue()))
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(main(["--target", "s600", "--dry-run"]), 2)


if __name__ == "__main__":
    unittest.main()
