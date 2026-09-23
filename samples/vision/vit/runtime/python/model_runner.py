# Copyright (c) 2026 D-Robotics Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Lazy board-runtime runner for a validated ViT model binding.

The machinery is shared (:mod:`samples._shared.model_runner`); this module
injects the ViT binding table so callers keep the pilot constructor
signature ``RuntimeModelRunner(selection)``.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from samples.vision.vit.runtime.python.model_binding import (
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
    """ViT runner: the shared lazy runner bound to the vit table."""

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
