"""Pinned official-source loading for numerical reference tests; no SDK calls."""
import hashlib
import importlib.util
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[4]
def load_source(path,name,*,base,hashes):
    archive=base/path
    if hashlib.sha256(archive.read_bytes()).hexdigest()!=hashes[path]:
        raise AssertionError(f"Fixed source bytes changed: {path}")
    spec=importlib.util.spec_from_file_location(name,archive)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module
    spec.loader.exec_module(module)
    return module
