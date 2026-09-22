"""Lazy board-runtime runner for a validated FastViT model binding.

The machinery is shared (:mod:`samples._shared.model_runner`); this module
injects the FastViT binding table so callers keep the pilot constructor
signature ``RuntimeModelRunner(selection)``.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from samples.vision.fastvit.runtime.python.model_binding import (
    BINDING_TABLE,
    ModelBinding,
    ModelSelection,
)
from samples._shared.model_runner import (  # noqa: F401 - re-exported surface
    RuntimeUnavailableError,
    _default_runtime_factory,
)
from samples._shared.model_runner import RuntimeModelRunner as _SharedRunner


class RuntimeModelRunner(_SharedRunner):
    """FastViT runner: the shared lazy runner bound to the fastvit table."""

    def __init__(
        self,
        selection: ModelSelection,
        *,
        runtime_factory: Optional[Callable[[str], Any]] = None,
        runtime: Any = None,
    ) -> None:
        super().__init__(
            selection,
            table=BINDING_TABLE,
            runtime_factory=runtime_factory,
            runtime=runtime,
        )


def create_runner(selection: ModelSelection, *,
                  runtime_factory: Optional[Callable[[str], Any]] = None,
                  runtime: Any = None) -> RuntimeModelRunner:
    """Construct a lazy runner for one resolved selection."""

    return RuntimeModelRunner(
        selection,
        runtime_factory=runtime_factory,
        runtime=runtime,
    )


__all__ = ["RuntimeModelRunner", "RuntimeUnavailableError", "create_runner"]
