"""Pinned access to the removed ``platforms/`` historical copies for tests.

The ``platforms/`` directory was removed from the active branch (see
the pinned historical record https://github.com/D-Robotics/rdk_model_zoo/blob/6c5cd7f2800bd0b341f2bb7b110d0a0454eefeba/docs/migration/2026-09-30-model-examples.md).  Its content stays
reachable through Git: ``PIN`` below is the last commit that touched the
tree and is reachable from the repository's ``develop`` history, so a
clone with full history has every object.  Tests that compare unified
code against the preserved platform sources — or load baseline
implementations — read them through this module instead of the worktree.

A shallow or partial clone that lacks the object gets an error naming
the exact fetch command::

    git fetch origin d2d2a4e0a898697bdfe5f68a9740a8c7d7cad57d

There is deliberately no "exists?" probe: source verification must go
through :func:`legacy_path`/:func:`legacy_tree`, which raise on missing
objects, so an absent historical file can never silently skip a check.

Materialized files land in a per-process temporary directory that is
removed at interpreter exit; nothing is written into the repository
worktree.
"""

from __future__ import annotations

import atexit
import shutil
import subprocess
import tempfile
from pathlib import Path

#: Repository root (this file lives at ``utils/py_utils/``).
ROOT = Path(__file__).resolve().parents[2]

#: The last commit that touched ``platforms/`` before its removal; its tree
#: is byte-identical to the removed worktree copy.
PIN = "d2d2a4e0a898697bdfe5f68a9740a8c7d7cad57d"

_TEMP = Path(tempfile.mkdtemp(prefix="rdk-model-zoo-platforms-pin-"))
atexit.register(shutil.rmtree, _TEMP, ignore_errors=True)


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(ROOT), *args], capture_output=True, check=False)


def legacy_path(relative: str) -> Path:
    """Return a local file for ``platforms/<relative>`` as pinned at ``PIN``.

    ``relative`` is the path below ``platforms/``, for example
    ``"x5/samples/vision/repvit/runtime/python/repvit.py"``.

    Raises:
        FileNotFoundError: When the object is not readable at the pin —
            including the shallow-clone case, with the exact fetch command.
    """

    repo_relative = f"platforms/{relative}"
    target = _TEMP / relative
    if target.is_file():
        return target
    result = _git("show", f"{PIN}:{repo_relative}")
    if result.returncode != 0:
        raise FileNotFoundError(
            f"{repo_relative} is not readable at pinned commit {PIN[:12]}. "
            f"A shallow clone must fetch the object first: "
            f"git fetch origin {PIN}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(result.stdout)
    return target


def legacy_tree(relative: str) -> Path:
    """Materialize the pinned directory ``platforms/<relative>`` and return it.

    Use for fixtures that need a whole subtree (for example a conversion
    directory); prefer :func:`legacy_path` for single files.

    Raises:
        FileNotFoundError: When the directory is not present at the pin, or
            any listed blob fails to materialize (never a silent skip).
    """

    # -z: paths come NUL-terminated and unquoted. Git's default ls-tree
    # output C-quotes non-ASCII names (the historical tree contains such
    # names), which would corrupt the materialized filenames.
    listing = _git("ls-tree", "-r", "-z", "--name-only", PIN, "--",
                   f"platforms/{relative}")
    if listing.returncode != 0 or not listing.stdout.strip():
        raise FileNotFoundError(
            f"platforms/{relative} is not present in the pinned tree {PIN[:12]}")
    target_root = _TEMP / relative
    for raw in listing.stdout.split(b"\0"):
        if not raw:
            continue
        repo_relative = raw.decode("utf-8")
        inside = repo_relative[len("platforms/"):]
        target = _TEMP / inside
        if target.is_file():
            continue
        blob = _git("show", f"{PIN}:{repo_relative}")
        if blob.returncode != 0:
            raise FileNotFoundError(
                f"{repo_relative} is not readable at pinned commit {PIN[:12]}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(blob.stdout)
    return target_root


def pinned_name(path) -> str:
    """Return the ``platforms/<relative>`` tree name of a materialized path."""

    # Resolve both sides: on macOS the symlinked tempdir (/var vs
    # /private/var) would otherwise break relative_to.
    resolved = Path(path).resolve()
    return f"platforms/{resolved.relative_to(_TEMP.resolve())}"


def legacy_module_namespace():
    """Register a ``platforms`` package over the materialized tree (once).

    Historical sources import each other under ``platforms.<group>...``
    module names, which used to resolve from the worktree.  Executing them
    from the pinned materialization needs that package root to point at the
    temporary tree instead.  The shared ``utils/py_utils`` helpers both
    groups import are materialized eagerly so the imports resolve.
    """

    import sys
    import types

    legacy_tree("x5/utils/py_utils")
    legacy_tree("s/utils/py_utils")
    module = sys.modules.get("platforms")
    if module is None or str(_TEMP) not in getattr(module, "__path__", ()):
        module = types.ModuleType("platforms")
        module.__path__ = [str(_TEMP)]
        sys.modules["platforms"] = module
    return module


__all__ = [
    "PIN",
    "ROOT",
    "legacy_module_namespace",
    "legacy_path",
    "legacy_tree",
    "pinned_name",
]
