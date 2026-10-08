"""Pinned official-source loading for numerical reference tests; no SDK calls.

``base`` paths under the removed ``platforms/`` tree resolve through the
pinned-commit helper, so the sha256 fixture still pins the same historical
bytes the worktree used to carry.
"""
import hashlib
import importlib.util
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[4]
from utils.py_utils.tests.legacy_platforms import legacy_path  # noqa: E402
def load_source(path,name,*,base,hashes):
    if base.name in ("x5","s") and base.parent.name=="platforms":
        archive=legacy_path(f"{base.name}/{path}")
    else:
        archive=base/path
    if hashlib.sha256(archive.read_bytes()).hexdigest()!=hashes[path]:
        raise AssertionError(f"Fixed source bytes changed: {path}")
    spec=importlib.util.spec_from_file_location(name,archive)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module
    spec.loader.exec_module(module)
    return module
