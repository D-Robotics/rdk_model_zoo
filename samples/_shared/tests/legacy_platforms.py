"""Test-import surface for the pinned historical-platforms reader.

The implementation lives in :mod:`samples._shared.legacy_platforms`
(shared with the evaluators); this module keeps the tests import path.
"""

from samples._shared.legacy_platforms import (  # noqa: F401 - re-exported surface
    PIN,
    ROOT,
    legacy_exists,
    legacy_module_namespace,
    legacy_path,
    legacy_tree,
    pinned_name,
)

__all__ = [
    "PIN",
    "ROOT",
    "legacy_exists",
    "legacy_module_namespace",
    "legacy_path",
    "legacy_tree",
    "pinned_name",
]
