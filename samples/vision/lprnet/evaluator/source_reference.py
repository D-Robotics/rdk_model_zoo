"""Load the fixed LPRNet source for migration comparison; not a runtime dependency."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]


def source_paths(target: str = "x5") -> tuple[Path, ...]:
    """Return the fixed-source files the comparison runs and hashes."""

    if target != "x5":
        raise ValueError("LPRNet has no fixed source outside X5.")
    base = ROOT / "platforms" / "x5"
    return (base / "samples" / "vision" / "lprnet" / "runtime" / "python" / "lprnet.py",)


def load_legacy(selection, factory, test_bin: str):
    """Instantiate the exact source wrapper with an injected recording SDK factory.

    Only the ``hbm_runtime`` module key is temporarily replaced for the host
    fixture; the source numeric code and decoding rule are untouched.
    """

    paths = source_paths("x5")

    def load(path: Path, name: str):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module

    old = sys.modules.get("hbm_runtime")
    if old is None:
        sys.modules["hbm_runtime"] = types.SimpleNamespace(HB_HBMRuntime=factory)
    try:
        module = load(paths[0], "lprnet_source_x5")
        # The source owns its imported SDK reference; do not mutate a real SDK.
        module.hbm_runtime = types.SimpleNamespace(HB_HBMRuntime=factory)
    finally:
        if old is None:
            sys.modules.pop("hbm_runtime", None)
        else:
            sys.modules["hbm_runtime"] = old
    config = module.LPRNetConfig(str(selection.model_path), str(test_bin))
    return module.LPRNet(config)
