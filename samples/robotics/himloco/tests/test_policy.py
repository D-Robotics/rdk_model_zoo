"""Offline policy math, not SDK or live robot validation."""

import unittest
from unittest.mock import Mock
import numpy as np
from samples.robotics.himloco.runtime.python.policy import HimLocoTask


class CanonicalStageTests(unittest.TestCase):
    """Readable-runtime canonical stage names on the offline policy task."""

    @staticmethod
    def make_task():
        actions = np.arange(12, dtype=np.float32).reshape(1, 12)
        calls = []

        def runner(feed):
            calls.append(feed)
            return {"actions": actions.copy()}

        return HimLocoTask(runner), calls

    def test_canonical_stages_match_legacy_aliases(self):
        task, _ = self.make_task()
        values = np.arange(270, dtype=np.float32)
        first = task.preprocess(values)
        second = task.pre_process(values)
        np.testing.assert_array_equal(
            first.tensors["obs_history"], second.tensors["obs_history"]
        )
        self.assertTrue(first.tensors["obs_history"].flags.c_contiguous)
        raw = task.infer(first.tensors)
        np.testing.assert_array_equal(
            raw.tensors["actions"], task.forward(second.tensors).tensors["actions"]
        )
        legacy = task.post_process(task.forward(first.tensors))
        np.testing.assert_array_equal(task.postprocess(raw).actions, legacy.actions)

    def test_predict_equals_explicit_canonical_chain(self):
        task, calls = self.make_task()
        values = np.arange(270, dtype=np.float32).reshape(6, 45)
        prepared = task.preprocess(values)
        explicit = task.postprocess(task.infer(prepared.tensors))
        result = task.predict(values)
        np.testing.assert_array_equal(explicit.actions, result.actions)
        # Latency measures each runner call separately; both stay valid.
        self.assertGreaterEqual(explicit.latency_ms, 0)
        self.assertGreaterEqual(result.latency_ms, 0)
        self.assertEqual(len(calls), 2)

    def test_consecutive_calls_keep_geometry_and_call_count(self):
        task, calls = self.make_task()
        first = task.predict(np.zeros(270))
        second = task.predict(np.ones(270))
        self.assertEqual(first.actions.shape, (1, 12))
        self.assertEqual(second.actions.shape, (1, 12))
        self.assertEqual(len(calls), 2)
        task.predict(np.full(270, 5.0))
        self.assertEqual(len(calls), 3)
        np.testing.assert_array_equal(
            task.preprocess(np.full(270, 7.0)).tensors["obs_history"][0, :270],
            np.full(270, 7.0, np.float32),
        )


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
