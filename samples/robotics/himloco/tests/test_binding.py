"""Exact publication and runtime transport checks using an explicit SDK double."""

from dataclasses import replace
from pathlib import Path
import unittest
from unittest.mock import Mock
import numpy as np
from samples.robotics.himloco.runtime.python.model_binding import (
    resolve_selection,
    bind_model,
    ASSET_ID,
)
from samples.robotics.himloco.runtime.python.policy import RuntimeModelRunner


def metadata():
    return dict(
        model_name="policy",
        model_names=["policy"],
        input_names=["obs_history"],
        input_shapes={"obs_history": [1, 270]},
        input_dtypes={"obs_history": "float32"},
        output_names=["actions"],
        output_shapes={"actions": [1, 12]},
        output_dtypes={"actions": "float32"},
    )


class BindingTests(unittest.TestCase):
    def test_publication_and_explicit_identity(self):
        selected = resolve_selection("x5")
        self.assertEqual(selected.asset.reference, ASSET_ID)
        self.assertEqual(selected.model_path.parent.name, "bayes-e")
        self.assertEqual(len(selected.asset.sha256), 64)
        with self.assertRaises(ValueError):
            resolve_selection("s100")
        with self.assertRaises(ValueError):
            resolve_selection("x5", model_path="/tmp/policy.bin")
        custom = resolve_selection(
            "x5", model_path="/tmp/policy.bin", asset_id=ASSET_ID
        )
        self.assertTrue(custom.explicit_model_path)

    def test_fixed_names_dtypes_shapes_and_single_model(self):
        selected = resolve_selection("x5")
        self.assertEqual(bind_model(selected, metadata()).input_name, "obs_history")
        for change in (
            {"input_names": ["wrong"]},
            {"output_shapes": {"actions": [12]}},
            {"output_dtypes": {"actions": "int32"}},
            {"model_names": ["policy", "other"]},
        ):
            with self.subTest(change=change), self.assertRaises(ValueError):
                bind_model(selected, {**metadata(), **change})

    def test_forged_selection_rejected_before_sdk_factory(self):
        factory = Mock()
        with self.assertRaises(ValueError):
            RuntimeModelRunner(
                replace(resolve_selection("x5"), model_path=Path("/tmp/unbound.bin")),
                runtime_factory=factory,
            )
        factory.assert_not_called()

    def test_real_path_board_gate_precedes_sdk_import(self):
        runner = RuntimeModelRunner(resolve_selection("x5"))
        runner._execution_target_gate = Mock(side_effect=ValueError("not x5"))
        with self.assertRaisesRegex(ValueError, "not x5"):
            runner.load()

    def test_injected_runtime_preserves_raw_actions_and_scheduling(self):
        from types import SimpleNamespace

        m = metadata()
        runtime = SimpleNamespace(model_names=m["model_names"])
        for field in (
            "input_names",
            "input_shapes",
            "input_dtypes",
            "output_names",
            "output_shapes",
            "output_dtypes",
        ):
            setattr(runtime, field, {"policy": m[field]})
        actions = np.arange(12, dtype=np.float32).reshape(1, 12)
        runtime.run = Mock(return_value={"policy": {"actions": actions}})
        runtime.set_scheduling_params = Mock()
        runner = RuntimeModelRunner(
            resolve_selection("x5"), runtime_factory=lambda path: runtime
        )
        runner.load()
        runner.set_scheduling_params(priority=3, bpu_cores=[0])
        runtime.set_scheduling_params.assert_called_once_with(
            priority={"policy": 3}, bpu_cores={"policy": [0]}
        )
        outputs = runner({"obs_history": np.zeros((1, 270), np.float32)})
        np.testing.assert_array_equal(outputs["actions"], actions)
        self.assertFalse(np.shares_memory(outputs["actions"], actions))


if __name__ == "__main__":
    unittest.main()
