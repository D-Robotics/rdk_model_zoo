"""Host tests for the thin board SDK session wrapper.

These tests pin the RuntimeSession call contract only: SDK-free
construction, the identity gate in front of the SDK factory, one
construction for repeated runs, native-mapping passthrough, and retry
after a failed load. The SDK factory and board identity are injected
seams; nothing here loads a real board SDK or certifies board inference.
"""

from __future__ import annotations

import sys
import unittest
from unittest import mock

import samples._shared.runtime as shared_runtime


class _FakeSdkModel:
    """Minimal SDK object recording the mappings it was called with."""

    def __init__(self):
        self.run_calls = []

    def run(self, inputs):
        self.run_calls.append(inputs)
        return {"count": len(self.run_calls)}


class RuntimeSessionConstructionTests(unittest.TestCase):
    def test_construction_never_touches_the_sdk_factory(self):
        factory = mock.Mock()

        with mock.patch.object(
            shared_runtime, "_default_runtime_factory", return_value=factory
        ):
            session = shared_runtime.RuntimeSession("/tmp/model.bin", target="x5")

        factory.assert_not_called()
        self.assertFalse(session.loaded)

    def test_construction_does_not_import_the_board_sdk(self):
        with mock.patch.dict(sys.modules, {"hbm_runtime": mock.Mock()}):
            sys.modules.pop("hbm_runtime", None)
            shared_runtime.RuntimeSession("/tmp/model.bin", target="x5")
        self.assertNotIn("hbm_runtime", sys.modules)

    def test_rejects_an_empty_model_path(self):
        with self.assertRaises(ValueError):
            shared_runtime.RuntimeSession("", target="x5")

    def test_runtime_property_requires_a_successful_load(self):
        session = shared_runtime.RuntimeSession("/tmp/model.bin", target="x5")

        with self.assertRaisesRegex(RuntimeError, r"load\(\) first"):
            _ = session.runtime


class RuntimeSessionIdentityGateTests(unittest.TestCase):
    def test_target_mismatch_fails_before_the_sdk_factory(self):
        factory = mock.Mock()

        with mock.patch.object(
            shared_runtime, "_default_runtime_factory", return_value=factory
        ), mock.patch(
            "samples._shared.platforms.detect_target", return_value="x5"
        ):
            session = shared_runtime.RuntimeSession("/tmp/model.bin", target="s100")
            with self.assertRaises(ValueError) as raised:
                session.load()

        self.assertIn("Target mismatch", str(raised.exception))
        self.assertIn("s100", str(raised.exception))
        self.assertIn("x5", str(raised.exception))
        factory.assert_not_called()
        self.assertFalse(session.loaded)

    def test_unrecognized_board_fails_before_the_sdk_factory(self):
        factory = mock.Mock()

        with mock.patch.object(
            shared_runtime, "_default_runtime_factory", return_value=factory
        ), mock.patch(
            "samples._shared.platforms.detect_target", return_value=None
        ):
            session = shared_runtime.RuntimeSession("/tmp/model.bin", target="x5")
            with self.assertRaises(ValueError) as raised:
                session.load()

        self.assertIn("recognized board identity", str(raised.exception))
        factory.assert_not_called()


class RuntimeSessionExecutionTests(unittest.TestCase):
    def test_repeated_runs_construct_the_model_once(self):
        model = _FakeSdkModel()
        factory = mock.Mock(return_value=model)

        with mock.patch.object(
            shared_runtime, "_default_runtime_factory", return_value=factory
        ), mock.patch(
            "samples._shared.platforms.detect_target", return_value="x5"
        ):
            session = shared_runtime.RuntimeSession("/tmp/model.bin", target="auto")
            session.run({"m": {"data": 1}})
            session.run({"m": {"data": 2}})

        factory.assert_called_once_with("/tmp/model.bin")

    def test_run_passes_native_mappings_through_unchanged(self):
        model = _FakeSdkModel()
        factory = mock.Mock(return_value=model)

        with mock.patch.object(
            shared_runtime, "_default_runtime_factory", return_value=factory
        ), mock.patch(
            "samples._shared.platforms.detect_target", return_value="x5"
        ):
            session = shared_runtime.RuntimeSession("/tmp/model.bin", target="x5")
            first = {"m": {"data": 1}}
            second = {"m": {"data": 2}}
            out_first = session.run(first)
            out_second = session.run(second)

        # The exact mapping objects reach the SDK and its exact return
        # values come back: no container adaptation in the shared layer.
        self.assertIs(model.run_calls[0], first)
        self.assertIs(model.run_calls[1], second)
        self.assertEqual(out_first, {"count": 1})
        self.assertEqual(out_second, {"count": 2})
        self.assertIs(session.runtime, model)

    def test_failed_load_keeps_no_success_state_and_can_retry(self):
        model = _FakeSdkModel()
        factory = mock.Mock(side_effect=[RuntimeError("construction boom"), model])

        with mock.patch.object(
            shared_runtime, "_default_runtime_factory", return_value=factory
        ), mock.patch(
            "samples._shared.platforms.detect_target", return_value="x5"
        ):
            session = shared_runtime.RuntimeSession("/tmp/model.bin", target="x5")

            with self.assertRaises(RuntimeError) as raised:
                session.load()
            self.assertIn("construction boom", str(raised.exception))
            self.assertFalse(session.loaded)
            with self.assertRaisesRegex(RuntimeError, r"load\(\) first"):
                _ = session.runtime

            session.load()

        self.assertTrue(session.loaded)
        self.assertIs(session.runtime, model)
        self.assertEqual(
            factory.call_args_list,
            [mock.call("/tmp/model.bin"), mock.call("/tmp/model.bin")],
        )

    def test_sdk_import_failure_preserves_the_original_cause(self):
        def missing_sdk():
            raise shared_runtime.RuntimeUnavailableError(
                "hbm_runtime is required for board execution."
            )

        with mock.patch.object(
            shared_runtime, "_default_runtime_factory", missing_sdk
        ), mock.patch(
            "samples._shared.platforms.detect_target", return_value="x5"
        ):
            session = shared_runtime.RuntimeSession("/tmp/model.bin", target="x5")
            with self.assertRaises(shared_runtime.RuntimeUnavailableError):
                session.run({"m": {}})

        self.assertFalse(session.loaded)


if __name__ == "__main__":
    unittest.main()
