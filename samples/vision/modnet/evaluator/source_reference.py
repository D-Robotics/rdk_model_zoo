"""Load the fixed MODNet source for migration comparison; not a runtime dependency."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]


def source_paths(target: str = "x5") -> tuple[Path, ...]:
    """Return the fixed-source files the comparison runs and hashes."""

    if target != "x5":
        raise ValueError("MODNet has no fixed source outside X5.")
    base = ROOT / "platforms" / "x5"
    return (base / "samples" / "vision" / "modnet" / "runtime" / "python" / "modnet.py",)


def load_legacy(selection, factory, *, ref_size: int = 512):
    """Instantiate the exact source wrapper with an injected recording SDK factory.

    Only the ``hbm_runtime`` module key is temporarily replaced for the host
    fixture.  The source imports ``utils.py_utils.inspect`` through a relative
    ``sys.path`` append, so the repository root is placed on ``sys.path`` for the
    duration of the load and restored afterwards.
    """

    paths = source_paths("x5")

    def load(path: Path, name: str):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module

    old_runtime = sys.modules.get("hbm_runtime")
    path_before = list(sys.path)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    if old_runtime is None:
        sys.modules["hbm_runtime"] = types.SimpleNamespace(HB_HBMRuntime=factory)
    try:
        module = load(paths[0], "modnet_source_x5")
        module.hbm_runtime = types.SimpleNamespace(HB_HBMRuntime=factory)
    finally:
        if old_runtime is None:
            sys.modules.pop("hbm_runtime", None)
        else:
            sys.modules["hbm_runtime"] = old_runtime
        sys.path[:] = path_before
    return module.MODNet(module.MODNetConfig(str(selection.model_path), ref_size=ref_size))
