# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Thin board SDK session: identity gate, lazy construction, passthrough run.

``RuntimeSession`` is the narrow shared wrapper around the board-side
``hbm_runtime`` SDK.  It owns exactly the board-side concerns every sample
repeats — importing the SDK, constructing one model instance, and executing
it — behind the exact-target identity check.  It is not an inference engine:
tensor names, input validation, output semantics and scheduling stay with
each sample's binding and runner.  Inputs and outputs keep the SDK's native
mappings and are passed through unchanged; this layer never guesses model
names or output roles.
"""

from __future__ import annotations

import importlib
import os
from typing import Any, Callable, Mapping, Optional


class RuntimeUnavailableError(RuntimeError):
    """The board runtime is unavailable in the current Python environment."""


def _default_runtime_factory() -> Callable[[str], Any]:
    """Import the board SDK lazily and return its model constructor."""

    try:
        runtime_module = importlib.import_module("hbm_runtime")
    except ImportError as exc:
        raise RuntimeUnavailableError(
            "hbm_runtime is required for board execution. Install/use the "
            "matching RDK Python environment; host help/list/dry-run do not need it."
        ) from exc
    factory = getattr(runtime_module, "HB_HBMRuntime", None)
    if not callable(factory):
        raise RuntimeUnavailableError(
            "Installed hbm_runtime does not expose HB_HBMRuntime."
        )
    return factory


class RuntimeSession:
    """One board SDK model instance behind an exact-target identity gate.

    Construction imports nothing board-side, so ``--help``, model listing
    and dry-run paths stay usable on a development host.  :meth:`load`
    performs the actual target check before the SDK is imported or the
    model is constructed: an explicit target that disagrees with the
    observed board fails without touching ``hbm_runtime``.  A failed load
    leaves no half-constructed state and can be retried.  No ``close`` or
    context-manager protocol is assumed; resource handling follows what
    the installed SDK actually documents.
    """

    def __init__(self, model_path: "str | os.PathLike[str]",
                 *, target: Optional[str]) -> None:
        path = os.fspath(model_path)
        if not str(path).strip():
            raise ValueError("model_path must be a non-empty path string.")
        self.model_path = str(path)
        #: Requested target (``auto``/None resolves to the detected board).
        self.target: Optional[str] = target
        self._runtime: Optional[Any] = None

    @property
    def loaded(self) -> bool:
        """Whether SDK model construction has succeeded."""

        return self._runtime is not None

    @property
    def runtime(self) -> Any:
        """Return the loaded SDK object; raises until :meth:`load` succeeds."""

        if self._runtime is None:
            raise RuntimeError("Runtime has not been loaded; call load() first.")
        return self._runtime

    def load(self) -> None:
        """Check the board identity, then construct the SDK model once."""

        if self._runtime is not None:
            return

        # Imported at call time so construction stays SDK-free and the gate
        # below runs for every caller of this class, not just module importers
        # that happened to pass through a checked entrypoint.
        from utils.py_utils.platforms import require_execution_target

        # The identity gate runs before the SDK import and model
        # construction: a requested/detected mismatch fails with zero SDK
        # factory calls.
        require_execution_target(self.target)
        try:
            self._runtime = _default_runtime_factory()(self.model_path)
        except Exception:
            # Preserve the original exception and leave no fake-success
            # state; a later load() retries from a clean slate.
            self._runtime = None
            raise

    def run(self, inputs: Mapping[str, Any]) -> Mapping[str, Any]:
        """Execute once with the SDK's native input/output mappings."""

        self.load()
        return self._runtime.run(inputs)


__all__ = ["RuntimeSession", "RuntimeUnavailableError"]
