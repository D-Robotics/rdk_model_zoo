"""Portable native host dependency discovery for maintainer host checks.

Shared by the native LLM host tests (and the host-validation runner) to
resolve the third-party pieces those checks compile against:

* nlohmann-json headers — ``json_include_dir``
* iconv linker flags — ``iconv_link_flags`` (``-liconv`` only where the
  platform keeps iconv outside libc, i.e. macOS)
* gflags compile/link flags — ``gflags_compile_flags``

Resolution order for every dependency: a documented explicit override first
(validated strictly — a wrong override is an error, never a silent skip),
then ``pkg-config``, then standard system install roots. There is no default
that reaches outside these channels, in particular no personal
``.coordination`` directory.

Explicit overrides:

* ``GEMMA_JSON_INCLUDE`` / ``MINICPM_JSON_INCLUDE`` — per-sample nlohmann
  include directory, passed by each sample's test module as the ``override``
  argument.
* ``GFLAGS_INCLUDE_DIR`` + ``GFLAGS_LIB_DIR`` — gflags install locations;
  both must be set together.

Errors:

* :class:`NativeDependencyMissing` — nothing found; host tests may surface
  this as an explicit skip reason (a missing prerequisite), while the CI
  runner installs the dependencies and rejects the skip.
* :class:`NativeDependencyOverrideError` — an explicitly provided override is
  invalid; this always fails the run.
"""

from __future__ import annotations

import os
from pathlib import Path
import platform
import shlex
import shutil
import subprocess

__all__ = [
    "NativeDependencyError",
    "NativeDependencyMissing",
    "NativeDependencyOverrideError",
    "gflags_compile_flags",
    "iconv_link_flags",
    "json_include_dir",
]

JSON_HEADER = Path("nlohmann") / "json.hpp"
GFLAGS_HEADER = Path("gflags") / "gflags.h"
# Exact linker input names for ``-lgflags``: the unversioned shared objects
# (distributions and Homebrew install them as symlinks to a versioned file)
# and the static archive — the names ld actually searches. Versioned-only
# files such as ``libgflags.2.3.dylib`` are NOT linkable via ``-lgflags``
# (verified: ``ld: library 'gflags' not found``), and neither is a directory
# or a dangling symlink carrying a linker name. The separate
# ``libgflags_nothreads.*`` build is deliberately not linkable.
GFLAGS_LIBRARY_NAMES = (
    "libgflags.so",
    "libgflags.dylib",
    "libgflags.a",
)

GFLAGS_INCLUDE_ENV = "GFLAGS_INCLUDE_DIR"
GFLAGS_LIB_ENV = "GFLAGS_LIB_DIR"

# Standard install roots for Linux distributions and Homebrew.
STANDARD_INCLUDE_ROOTS = (
    "/usr/include",
    "/usr/local/include",
    "/opt/homebrew/include",
)
STANDARD_LIB_ROOTS = (
    "/usr/lib/x86_64-linux-gnu",
    "/usr/lib/aarch64-linux-gnu",
    "/usr/lib64",
    "/usr/lib",
    "/usr/local/lib",
    "/opt/homebrew/lib",
)

JSON_INSTALL_HINT = (
    "install nlohmann-json (Ubuntu: nlohmann-json3-dev, Homebrew: nlohmann-json)"
    " or point GEMMA_JSON_INCLUDE/MINICPM_JSON_INCLUDE at a directory"
    " containing nlohmann/json.hpp"
)
GFLAGS_INSTALL_HINT = (
    "install gflags (Ubuntu: libgflags-dev, Homebrew: gflags) or set"
    f" {GFLAGS_INCLUDE_ENV} and {GFLAGS_LIB_ENV} together"
)


class NativeDependencyError(Exception):
    """Base error for native host dependency resolution problems."""


class NativeDependencyMissing(NativeDependencyError):
    """A native dependency is not installed or discoverable on this host."""


class NativeDependencyOverrideError(NativeDependencyError):
    """An explicitly provided dependency override is invalid."""


def _pkg_config_output(package, mode, environ):
    """Run pkg-config for ``package`` (``cflags``/``libs``); None if unusable."""
    binary = shutil.which("pkg-config", path=environ.get("PATH"))
    if binary is None:
        return None
    env = dict(os.environ)
    for key, value in environ.items():
        if isinstance(value, str):
            env[key] = value
    done = subprocess.run(
        [binary, f"--{mode}", package],
        capture_output=True,
        text=True,
        env=env,
    )
    if done.returncode != 0:
        return None
    return done.stdout


def _pkg_config_tokens(package, mode, environ, pkg_config):
    if pkg_config is None:
        pkg_config = _pkg_config_output
    try:
        output = pkg_config(package, mode, environ)
    except (OSError, subprocess.SubprocessError):
        return None
    if output is None:
        return None
    return shlex.split(output)


def _normalized_include_tokens(tokens):
    """Canonicalize ``-Idir`` / ``-I dir`` tokens into ``['-I', dir]`` pairs.

    Other tokens (e.g. ``-DNDEBUG``) pass through unchanged.
    """
    normalized = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token == "-I" and index + 1 < len(tokens):
            normalized.extend(("-I", tokens[index + 1]))
            index += 2
        elif token.startswith("-I") and len(token) > 2:
            normalized.extend(("-I", token[2:]))
            index += 1
        else:
            normalized.append(token)
            index += 1
    return normalized


def _include_dirs_from_tokens(tokens):
    """Extract include directories from pkg-config cflag tokens."""
    normalized = _normalized_include_tokens(tokens)
    return [
        Path(normalized[index + 1])
        for index in range(len(normalized) - 1)
        if normalized[index] == "-I"
    ]


def _explicit_override(override, label):
    if isinstance(override, Path):
        candidate = override.expanduser()
    else:
        text = str(override).strip()
        if not text:
            return None
        candidate = Path(text).expanduser()
    if not candidate.is_dir():
        raise NativeDependencyOverrideError(
            f"{label} override is not a directory: {candidate}"
        )
    return candidate


def json_include_dir(override=None, *, environ=None, pkg_config=None,
                     include_roots=None):
    """Resolve the nlohmann-json include directory and return it as a Path.

    ``override`` is a per-sample explicit include directory (the sample test
    modules pass their ``GEMMA_JSON_INCLUDE``/``MINICPM_JSON_INCLUDE``
    environment values). A non-empty override must be a directory containing
    ``nlohmann/json.hpp``; otherwise :class:`NativeDependencyOverrideError`
    is raised. Without an override, discovery checks ``pkg-config
    nlohmann_json`` cflags and then the standard system include roots.
    :class:`NativeDependencyMissing` is raised when nothing resolves.

    ``environ``, ``pkg_config`` and ``include_roots`` are injection points for
    hermetic tests and default to the process environment, the real
    pkg-config and the standard roots.
    """
    environ = os.environ if environ is None else environ
    if override is not None:
        candidate = _explicit_override(override, "nlohmann-json include")
        if candidate is not None:
            if not (candidate / JSON_HEADER).is_file():
                raise NativeDependencyOverrideError(
                    f"nlohmann-json include override has no {JSON_HEADER}:"
                    f" {candidate}"
                )
            return candidate
    roots = STANDARD_INCLUDE_ROOTS if include_roots is None else include_roots
    tokens = _pkg_config_tokens("nlohmann_json", "cflags", environ, pkg_config)
    if tokens:
        for directory in _include_dirs_from_tokens(tokens):
            if (directory / JSON_HEADER).is_file():
                return directory
        # A pkg-config entry without a usable header (e.g. empty cflags for
        # default-path installs) falls through to the standard roots.
    for root in roots:
        candidate = Path(root)
        if (candidate / JSON_HEADER).is_file():
            return candidate
    raise NativeDependencyMissing(
        f"nlohmann-json headers not found ({JSON_HEADER}); {JSON_INSTALL_HINT}"
    )


def iconv_link_flags(system=None):
    """Return the iconv linker flags for a host system.

    ``system`` defaults to ``platform.system()`` (case-insensitive). macOS
    keeps iconv in libiconv (``['-liconv']``); Linux carries it in libc
    (``[]``). Any other system raises :class:`NativeDependencyError` rather
    than guessing link flags.
    """
    name = platform.system().lower() if system is None else str(system).lower()
    if name == "darwin":
        return ["-liconv"]
    if name == "linux":
        return []
    raise NativeDependencyError(f"unsupported host system for iconv linking: {name!r}")


def _is_linkable_library(candidate):
    """A real readable linker input: not a directory, not a dangling symlink.

    ``Path.is_file`` follows symlinks, so a broken link or a link to a
    directory is rejected just like a directory named after a library.
    """
    return candidate.is_file() and os.access(candidate, os.R_OK)


def _gflags_library_exists(lib_dir):
    return any(
        _is_linkable_library(lib_dir / name) for name in GFLAGS_LIBRARY_NAMES
    )


def gflags_compile_flags(*, environ=None, pkg_config=None, include_roots=None,
                         lib_roots=None):
    """Resolve gflags flags as a ``(compile_flags, link_flags)`` tuple.

    Discovery order: the explicit ``GFLAGS_INCLUDE_DIR`` + ``GFLAGS_LIB_DIR``
    environment overrides (both required together, both validated), then
    ``pkg-config gflags``, then the standard system include/library roots.
    The returned flags splice directly into a compiler command, e.g.
    ``(["-I", "/usr/include"], ["-L", "/usr/lib", "-lgflags"])``.
    :class:`NativeDependencyMissing` is raised when gflags cannot be found.

    ``environ``, ``pkg_config``, ``include_roots`` and ``lib_roots`` are
    injection points for hermetic tests.
    """
    environ = os.environ if environ is None else environ
    include_env = str(environ.get(GFLAGS_INCLUDE_ENV, "")).strip()
    lib_env = str(environ.get(GFLAGS_LIB_ENV, "")).strip()
    if include_env or lib_env:
        if not (include_env and lib_env):
            raise NativeDependencyOverrideError(
                f"{GFLAGS_INCLUDE_ENV} and {GFLAGS_LIB_ENV} must be set together"
            )
        include_dir = _explicit_override(include_env, "gflags include")
        lib_dir = _explicit_override(lib_env, "gflags library")
        if not (include_dir / GFLAGS_HEADER).is_file():
            raise NativeDependencyOverrideError(
                f"gflags include override has no {GFLAGS_HEADER}: {include_dir}"
            )
        if not _gflags_library_exists(lib_dir):
            names = "/".join(GFLAGS_LIBRARY_NAMES)
            raise NativeDependencyOverrideError(
                f"gflags library override has no linkable {names}"
                f" (versioned-only files and unresolved symlinks do not"
                f" satisfy -lgflags) in: {lib_dir}"
            )
        return ["-I", str(include_dir)], ["-L", str(lib_dir), "-lgflags"]
    cflags = _pkg_config_tokens("gflags", "cflags", environ, pkg_config)
    libs = _pkg_config_tokens("gflags", "libs", environ, pkg_config)
    if cflags is not None and libs is not None:
        return _normalized_include_tokens(cflags), libs
    roots = STANDARD_INCLUDE_ROOTS if include_roots is None else include_roots
    libraries = STANDARD_LIB_ROOTS if lib_roots is None else lib_roots
    include_dir = next(
        (root for root in map(Path, roots) if (root / GFLAGS_HEADER).is_file()),
        None,
    )
    lib_dir = next(
        (root for root in map(Path, libraries) if _gflags_library_exists(root)),
        None,
    )
    if include_dir is None or lib_dir is None:
        missing = (
            "gflags headers" if include_dir is None else "a linkable gflags library"
        )
        raise NativeDependencyMissing(f"{missing} not found; {GFLAGS_INSTALL_HINT}")
    return ["-I", str(include_dir)], ["-L", str(lib_dir), "-lgflags"]
