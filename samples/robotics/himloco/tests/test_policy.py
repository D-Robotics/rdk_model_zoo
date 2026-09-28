"""Offline policy math, not SDK or live robot validation."""

import unittest
from unittest.mock import Mock
import numpy as np
from samples.robotics.himloco.runtime.python.policy import HimLocoTask


class PolicyTests(unittest.TestCase):
    def test_predict_matches_explicit_steps(self):
        runner = Mock(
            return_value={"actions": np.arange(12, dtype=np.float32).reshape(1, 12)}
        )
        task = HimLocoTask(runner)
        values = np.arange(270, dtype=np.float32)
        prepared = task.pre_process(values)
        explicit = task.post_process(task.forward(prepared.tensors))
        result = task.predict(values)
        np.testing.assert_array_equal(explicit.actions, result.actions)
        self.assertGreaterEqual(result.latency_ms, 0)

    def test_raw_actions_are_not_clipped_scaled_or_normalized(self):
        actions = np.linspace(-20, 20, 12, dtype=np.float32).reshape(1, 12)
        task = HimLocoTask(lambda feed: {"actions": actions})
        raw = task.forward(task.pre_process(np.zeros(270)).tensors)
        np.testing.assert_array_equal(raw.tensors["actions"], actions)
        result = task.post_process(raw)
        np.testing.assert_array_equal(result.actions, actions)
        actions.fill(0)
        self.assertNotEqual(float(result.actions[0, 0]), 0)

    def test_preparation_owns_data_and_preserves_history_order(self):
        source = np.arange(270, dtype=np.float32).reshape(6, 45)
        task = HimLocoTask(lambda feed: {})
        prepared = task.pre_process(source)
        source.fill(9)
        np.testing.assert_array_equal(
            prepared.tensors["obs_history"],
            np.arange(270, dtype=np.float32).reshape(1, 270),
        )
        self.assertTrue(prepared.tensors["obs_history"].flags.c_contiguous)

    def test_raw_results_do_not_share_last_call_latency(self):
        task = HimLocoTask(lambda feed: {"actions": np.zeros((1, 12), np.float32)})
        feed = task.pre_process(np.zeros(270)).tensors
        first = task.forward(feed)
        second = task.forward(feed)
        self.assertEqual(task.post_process(first).latency_ms, first.latency_ms)
        self.assertEqual(task.post_process(second).latency_ms, second.latency_ms)

    def test_invalid_inputs_fail_before_runner(self):
        runner = Mock()
        task = HimLocoTask(runner)
        for value in (
            np.zeros(269),
            np.full(270, np.nan),
            np.ones(270, np.complex64),
            ["1"] * 270,
        ):
            with self.subTest(dtype=np.asarray(value).dtype), self.assertRaises(
                ValueError
            ):
                task.predict(value)
        runner.assert_not_called()

    def test_raw_output_contract_is_exact_without_dtype_coercion(self):
        for outputs in (
            {"actions": np.zeros(12, np.float32)},
            {"actions": np.zeros((1, 12), np.float64)},
            {"actions": np.full((1, 12), np.inf, np.float32)},
            {"wrong": np.zeros((1, 12), np.float32)},
        ):
            with self.subTest(outputs=list(outputs)), self.assertRaises(ValueError):
                HimLocoTask(lambda feed: outputs).predict(np.zeros(270))


if __name__ == "__main__":
    unittest.main()
