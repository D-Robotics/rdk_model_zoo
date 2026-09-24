# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0

"""Board-output binding contract for LPRNet (X5 evidence 2026-09-24).

The released ``lpr.bin`` reports native logits metadata ``(1, 68, 18, 1)`` —
the measured board protocol.  The 3D ``(1, 68, 18)`` layout is the old
unified-contract/host-fixture shape kept as an explicit API-compatibility
binding for existing host tests and injected runners; no published SDK
artifact has been observed with it.  These tests pin the remediated
protocol: the binding records the complete native shape, ``forward`` keeps
it, and only ``post_process`` removes singleton axes before the source CTC
decode.
"""

from __future__ import annotations

import contextlib
import importlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np


BINDING = "samples.vision.lprnet.runtime.python.model_binding"
RUNNER = "samples.vision.lprnet.runtime.python.model_runner"
TASK = "samples.vision.lprnet.runtime.python.lprnet"
MAIN = "samples.vision.lprnet.runtime.python.main"
RUNTIME_META = "samples._shared.runtime_meta"

BOARD_OUTPUT_SHAPE = (1, 68, 18, 1)
LEGACY_API_OUTPUT_SHAPE = (1, 68, 18)
INPUT_SHAPE = (1, 3, 24, 94)


def _logits(shape):
    """One float32 logits array whose CTC payload decodes to 京A7."""

    value = np.full(shape, -6.0, dtype=np.float32)

    def cell(char, time):
        return (0, char, time) + ((0,) if value.ndim == 4 else ())

    for time, char in ((0, 0), (1, 41), (2, 38)):
        value[cell(char, time)] = 6.0
    # The blank label must win every remaining time step.
    value[cell(67, slice(3, None))] = 6.0
    return value


def make_runtime(output_shape, run_shape=None):
    """A fake ``hbm_runtime`` whose metadata reports ``output_shape``.

    ``run_shape`` lets a test return arrays that drift from the bound
    metadata shape after binding.
    """

    class BoardRuntime:
        model_names = ["lpr"]
        input_names = {"lpr": ["input"]}
        input_shapes = {"lpr": {"input": INPUT_SHAPE}}
        input_dtypes = {"lpr": {"input": "float32"}}
        output_names = {"lpr": ["output"]}
        output_shapes = {"lpr": {"output": output_shape}}
        input_strides = {"lpr": {}}
        output_strides = {"lpr": {}}
        output_dtypes = {"lpr": {"output": "float32"}}
        output_quants = {"lpr": {}}

        def __init__(self, path=None):
            self.run_shape = run_shape
            self.calls = []

        def run(self, tensors):
            self.calls.append(tensors)
            return {"lpr": {"output": _logits(self.run_shape or output_shape)}}

        def set_scheduling_params(self, **kwargs):
            self.scheduling = kwargs

    return BoardRuntime


def _metadata(output_names, output_shapes):
    return {
        "model_names": ("lpr",),
        "model_name": "lpr",
        "input_names": ("input",),
        "input_shapes": {"input": INPUT_SHAPE},
        "input_dtypes": {"input": "float32"},
        "output_names": output_names,
        "output_shapes": output_shapes,
        "output_dtypes": {name: "float32" for name in output_names},
    }


class BoardOutputBindingTests(unittest.TestCase):
    def test_binding_records_full_native_shape_of_released_artifact(self):
        binding_mod = importlib.import_module(BINDING)
        binding = binding_mod.bind_model(
            binding_mod.resolve_selection("x5"),
            _metadata(("output",), {"output": BOARD_OUTPUT_SHAPE}),
        )
        self.assertEqual(binding.output_shape, BOARD_OUTPUT_SHAPE)

    def test_legacy_api_3d_contract_binds_as_reported(self):
        binding_mod = importlib.import_module(BINDING)
        binding = binding_mod.bind_model(
            binding_mod.resolve_selection("x5"),
            _metadata(("output",), {"output": LEGACY_API_OUTPUT_SHAPE}),
        )
        self.assertEqual(binding.output_shape, LEGACY_API_OUTPUT_SHAPE)

    def test_binding_rejects_wrong_rank_and_axis_orders(self):
        binding_mod = importlib.import_module(BINDING)
        selection = binding_mod.resolve_selection("x5")
        for shape in (
            (1, 18, 68, 1),  # swapped classes/timesteps
            (68, 18),  # payload without batch axis
            (1, 68, 18, 2),  # rank 4 without a trailing singleton
            (1, 1, 68, 18),  # singleton axis is not trailing
        ):
            with self.subTest(shape=shape):
                with self.assertRaises(binding_mod.MetadataMismatchError):
                    binding_mod.bind_model(
                        selection, _metadata(("output",), {"output": shape})
                    )

    def test_binding_rejects_second_output_tensor(self):
        binding_mod = importlib.import_module(BINDING)
        with self.assertRaises(binding_mod.MetadataMismatchError):
            binding_mod.bind_model(
                binding_mod.resolve_selection("x5"),
                _metadata(("output", "aux"), {"output": BOARD_OUTPUT_SHAPE, "aux": (1, 4)}),
            )


class BoardOutputTaskTests(unittest.TestCase):
    def _task(self, output_shape, run_shape=None):
        binding_mod = importlib.import_module(BINDING)
        runner_mod = importlib.import_module(RUNNER)
        task_mod = importlib.import_module(TASK)
        runner = runner_mod.RuntimeModelRunner(
            binding_mod.resolve_selection("x5"),
            runtime=make_runtime(output_shape, run_shape)(),
        )
        binding = runner.load()
        return task_mod.LPRNetTask(runner, binding)

    def _input_dat(self):
        if not hasattr(self, "_dat"):
            handle = tempfile.TemporaryDirectory()
            self.addCleanup(handle.cleanup)
            self._dat = Path(handle.name) / "input.dat"
            np.arange(1 * 3 * 24 * 94, dtype=np.float32).reshape(INPUT_SHAPE).tofile(self._dat)
        return self._dat

    def test_forward_keeps_native_shape_and_post_process_decodes(self):
        task = self._task(BOARD_OUTPUT_SHAPE)
        prepared = task.pre_process(self._input_dat())
        raw = task.forward(prepared.tensors)
        self.assertEqual(raw.shape, BOARD_OUTPUT_SHAPE)
        self.assertEqual(raw.dtype, np.float32)
        self.assertEqual(task.post_process(raw), "京A7")
        self.assertEqual(task.predict(self._input_dat()), "京A7")

    def test_legacy_api_3d_contract_runs_end_to_end(self):
        task = self._task(LEGACY_API_OUTPUT_SHAPE)
        prepared = task.pre_process(self._input_dat())
        raw = task.forward(prepared.tensors)
        self.assertEqual(raw.shape, LEGACY_API_OUTPUT_SHAPE)
        self.assertEqual(task.predict(self._input_dat()), "京A7")

    def test_ctc_logits_reduces_only_singletons(self):
        task_mod = importlib.import_module(TASK)
        self.assertEqual(task_mod.ctc_logits(_logits(BOARD_OUTPUT_SHAPE)).shape, (68, 18))
        self.assertEqual(task_mod.ctc_logits(_logits(LEGACY_API_OUTPUT_SHAPE)).shape, (68, 18))
        with self.assertRaises(ValueError):
            task_mod.ctc_logits(np.zeros((1, 18, 68, 1), dtype=np.float32))

    def test_runner_rejects_shape_drift_after_binding(self):
        for bound, returned in (
            (BOARD_OUTPUT_SHAPE, LEGACY_API_OUTPUT_SHAPE),
            (LEGACY_API_OUTPUT_SHAPE, BOARD_OUTPUT_SHAPE),
        ):
            with self.subTest(bound=bound, returned=returned):
                task = self._task(bound, run_shape=returned)
                runtime_meta = importlib.import_module(RUNTIME_META)
                with self.assertRaises(runtime_meta.MetadataMismatchError):
                    task.predict(self._input_dat())

    def test_post_process_rejects_mismatched_raw(self):
        task = self._task(BOARD_OUTPUT_SHAPE)
        for raw in (
            _logits(LEGACY_API_OUTPUT_SHAPE),  # drifted shape (legacy API layout)
            np.zeros((1, 18, 68, 1), dtype=np.float32),  # permuted axes
            _logits(BOARD_OUTPUT_SHAPE).astype(np.float64),  # drifted dtype
        ):
            with self.subTest(shape=raw.shape, dtype=raw.dtype):
                with self.assertRaises(ValueError):
                    task.post_process(raw)

    def test_dry_run_describes_released_native_protocol(self):
        main = importlib.import_module(MAIN)
        with mock.patch.object(main, "RuntimeModelRunner", side_effect=AssertionError):
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                self.assertEqual(main.main(["--dry-run", "--target", "x5"]), 0)
        payload = json.loads(buffer.getvalue())
        self.assertEqual(payload["input_shape"], [1, 3, 24, 94])
        self.assertEqual(payload["input_dtype"], "float32")
        self.assertEqual(payload["output_shape"], [1, 68, 18, 1])
        self.assertEqual(payload["output_dtype"], "float32")
        self.assertIn("(1, 68, 18, 1)", payload["output_layout"])
        self.assertIn("(1, 68, 18)", payload["output_layout"])
        self.assertIn("(68, 18)", payload["output_layout"])


if __name__ == "__main__":
    unittest.main()
