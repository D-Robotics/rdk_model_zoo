"""Load the fixed YOLOWorld source for migration comparison; not a runtime dependency."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]


def source_paths(target: str = "x5") -> tuple[Path, ...]:
    """Return the fixed-source files the comparison runs and hashes."""

    if target != "x5":
        raise ValueError("YOLOWorld has no fixed source outside X5.")
    base = ROOT / "platforms" / "x5"
    return (
        base / "samples" / "vision" / "yoloworld" / "runtime" / "python" / "yoloworld_det.py",
        base / "utils" / "py_utils" / "postprocess.py",
    )


def load_legacy(selection, factory, vocab_file, *, score_thres: float, nms_thres: float):
    """Instantiate the exact source wrapper with an injected recording SDK factory.

    Only the ``hbm_runtime`` module key is temporarily replaced for the host
    fixture.  The source imports ``utils.py_utils.postprocess`` through a relative
    ``sys.path`` append, so the repository root is placed on ``sys.path`` for the
    duration of the load and restored afterwards; the shared postprocess module
    is loaded explicitly rather than relying on the caller's environment.
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
    # The source and its ``utils.py_utils.postprocess`` helper both import
    # ``hbm_runtime`` at module level, so the injected stub must expose both
    # names the source files bind.  The stub is installed for the duration of the
    # load and the caller's entry is restored afterwards.
    stub = types.SimpleNamespace(QuantParams=object, HB_HBMRuntime=factory)
    sys.modules["hbm_runtime"] = stub
    try:
        module = load(paths[0], "yoloworld_source_x5")
        module.hbm_runtime = stub
        # Bind the fixed platform postprocess explicitly instead of relying on
        # whatever ``utils.py_utils`` the caller's environment happens to expose.
        module.post_utils = load(paths[1], "platforms.x5.utils.py_utils.postprocess")
    finally:
        if old_runtime is None:
            sys.modules.pop("hbm_runtime", None)
        else:
            sys.modules["hbm_runtime"] = old_runtime
        sys.path[:] = path_before
    config = module.YOLOWorldConfig(
        str(selection.model_path), str(vocab_file), score_thres=score_thres, nms_thres=nms_thres
    )
    return module.YOLOWorldDetect(config)
