# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Capture a same-board FCOS source/unified comparison with complete evidence.

The evaluator intentionally requires a concrete X5 identity and a new output
directory. It never downloads an artifact. ``runtime_factory`` and
``legacy_runner_factory`` are host-only test seams; normal CLI execution uses
the board SDK and the fixed source wrapper.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import traceback
import types
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
SAMPLE_DIR = ROOT / "samples" / "vision" / "fcos"
SOURCE_FILE = ROOT / "platforms" / "x5" / "samples" / "vision" / "fcos" / "runtime" / "python" / "fcos_det.py"
SOURCE_REF = "ac115717197920355fc390bb04299b20e6436864"
_FIXED_SOURCE_MODULES: dict[str, types.ModuleType] = {}
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from samples._shared.platforms import require_execution_target  # noqa: E402
from samples._shared.runtime_meta import RuntimeMetadata  # noqa: E402
from samples.vision.fcos.runtime.python.fcos import FCOSTask  # noqa: E402
from samples.vision.fcos.runtime.python.model_binding import (  # noqa: E402
    ModelSelection,
    resolve_selection,
)
from samples.vision.fcos.runtime.python.model_runner import RuntimeModelRunner  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "quant_type"):
        quant_type = value.quant_type
        return {
            "quant_type": str(getattr(quant_type, "name", quant_type)),
            "scale": _jsonable(getattr(value, "scale", None)),
            "zero_point": _jsonable(getattr(value, "zero_point", None)),
            "axis": getattr(value, "axis", None),
        }
    if hasattr(value, "__dict__") and not isinstance(value, type):
        return {str(key): _jsonable(item) for key, item in vars(value).items()}
    return value


def _metadata_json(metadata: RuntimeMetadata) -> dict[str, Any]:
    return _jsonable(asdict(metadata))


def _load_fixed_source_helpers() -> None:
    """Load source preprocess/postprocess by absolute fixed-source paths."""
    if _FIXED_SOURCE_MODULES:
        sys.modules.update(_FIXED_SOURCE_MODULES)
        return
    source_root = SOURCE_FILE.parents[5]
    utils_root = source_root / "utils"
    py_utils_root = utils_root / "py_utils"
    utils_package = types.ModuleType("utils")
    utils_package.__path__ = [str(utils_root)]
    py_utils_package = types.ModuleType("utils.py_utils")
    py_utils_package.__path__ = [str(py_utils_root)]
    sys.modules["utils"] = utils_package
    sys.modules["utils.py_utils"] = py_utils_package
    _FIXED_SOURCE_MODULES.update({"utils": utils_package, "utils.py_utils": py_utils_package})
    for name in ("preprocess", "postprocess"):
        module_name = f"utils.py_utils.{name}"
        path = py_utils_root / f"{name}.py"
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Cannot load fixed source helper: {path}")
        module = importlib.util.module_from_spec(spec)
        module._FCOS_FIXED_SOURCE_HELPER = True
        sys.modules[module_name] = module
        _FIXED_SOURCE_MODULES[module_name] = module
        setattr(py_utils_package, name, module)
        spec.loader.exec_module(module)


def _save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(value), ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8")


def _save_side(side_root: Path, side: Mapping[str, Any], arrays: dict[str, Any]) -> None:
    side_name = side_root.name
    metadata_path = side_root / "metadata.json"
    result_path = side_root / "result.json"
    _save_json(metadata_path, side["metadata"])
    _save_json(result_path, side["result"])
    for path, category in ((metadata_path, "metadata"), (result_path, "result")):
        arrays[str(path.relative_to(side_root.parent))] = {
            "side": side_name,
            "category": category,
            "sha256": _sha256(path),
        }
    for category in ("inputs", "raw"):
        for name, value in side[category].items():
            path = side_root / category / f"{name}.npy"
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, np.asarray(value), allow_pickle=False)
            arrays[str(path.relative_to(side_root.parent))] = {
                "side": side_name,
                "category": category,
                "tensor_name": name,
                "shape": list(np.asarray(value).shape),
                "dtype": str(np.asarray(value).dtype),
                "sha256": _sha256(path),
            }


def _record_error(summary: dict[str, Any], label: str, exc: BaseException) -> None:
    summary.setdefault("errors", {})[label] = {
        "type": type(exc).__name__,
        "message": str(exc),
        "traceback": traceback.format_exc(),
    }


def _run_source(
    selection: ModelSelection,
    image: np.ndarray,
    *,
    resize_type: int,
    conf_thres: float,
    iou_thres: float,
    priority: int,
    bpu_cores: list[int],
) -> dict[str, Any]:
    """Run the fixed source wrapper and retain its raw runtime arrays."""
    if resize_type != 0:
        raise ValueError("FCOS source comparison only supports its published direct-resize mode (resize_type=0).")
    source_root = SOURCE_FILE.parents[5]
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    _load_fixed_source_helpers()
    module_name = "fcos_fixed_source_evaluator"
    spec = importlib.util.spec_from_file_location(module_name, SOURCE_FILE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load fixed FCOS source: {SOURCE_FILE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    source = module.FCOSDetect(module.FCOSConfig(
        str(selection.model_path),
        classes_num=selection.contract.classes_num,
        conf_thres=conf_thres,
        iou_thres=iou_thres,
        resize_type=resize_type,
    ))
    source.set_scheduling_params(priority=priority, bpu_cores=bpu_cores)
    inputs = source.pre_process(image, resize_type=resize_type)
    outputs = source.forward(inputs)
    result = source.post_process(outputs, image.shape[1], image.shape[0], conf_thres, iou_thres)
    flat_inputs = inputs[source.model_name]
    flat_outputs = outputs[source.model_name]
    return {
        "metadata": _metadata_json(RuntimeMetadata.from_runtime(source.model)),
        "inputs": {name: np.asarray(value).copy() for name, value in flat_inputs.items()},
        "raw": {name: np.asarray(value).copy() for name, value in flat_outputs.items()},
        "result": {"boxes": result[0], "scores": result[1], "class_ids": result[2]},
    }


def _run_unified(
    selection: ModelSelection,
    image: np.ndarray,
    *,
    resize_type: int,
    conf_thres: float,
    iou_thres: float,
    priority: int,
    bpu_cores: list[int],
    runtime_factory: Callable[[str], Any],
) -> dict[str, Any]:
    runtime = runtime_factory(str(selection.model_path))
    runner = RuntimeModelRunner(selection, runtime=runtime)
    binding = runner.load()
    runner.set_scheduling_params(priority=priority, bpu_cores=bpu_cores)
    task = FCOSTask(runner, binding, conf_thres=conf_thres, iou_thres=iou_thres, resize_type=resize_type)
    prepared = task.pre_process(image)
    outputs = task.forward(prepared)
    result = task.post_process(outputs, prepared.context)
    return {
        "metadata": _metadata_json(runner.metadata),
        "inputs": {name: np.asarray(value).copy() for name, value in prepared.tensors.items()},
        "raw": {name: np.asarray(value).copy() for name, value in outputs.items()},
        "result": {"boxes": result.boxes, "scores": result.scores, "class_ids": result.class_ids},
    }


def _compare_arrays(left: Mapping[str, Any], right: Mapping[str, Any], prefix: str) -> dict[str, bool]:
    checks: dict[str, bool] = {f"{prefix}.names": set(left) == set(right)}
    for name in sorted(set(left) & set(right)):
        a = np.asarray(left[name])
        b = np.asarray(right[name])
        checks[f"{prefix}.{name}"] = bool(a.shape == b.shape and a.dtype == b.dtype and np.array_equal(a, b))
    return checks


def _normalise_side(side: Mapping[str, Any]) -> dict[str, Any]:
    """Normalise the host test seam to the same evidence shape as the board path."""
    metadata = side["metadata"]
    return {
        "metadata": _metadata_json(metadata) if isinstance(metadata, RuntimeMetadata) else _jsonable(metadata),
        "inputs": {str(name): np.asarray(value).copy() for name, value in side["inputs"].items()},
        "raw": {str(name): np.asarray(value).copy() for name, value in side["raw"].items()},
        "result": {str(name): np.asarray(value).copy() for name, value in side["result"].items()},
    }


def run_comparison(
    selection: ModelSelection,
    image: np.ndarray,
    image_path: str | Path,
    output_dir: str | Path,
    *,
    resize_type: int = 0,
    conf_thres: float = 0.5,
    iou_thres: float = 0.6,
    priority: int = 0,
    bpu_cores: list[int] | None = None,
    runtime_factory: Callable[[str], Any] | None = None,
    legacy_runner_factory: Callable[..., dict[str, Any]] | None = None,
    command: list[str] | None = None,
) -> dict[str, Any]:
    """Run both implementations and retain raw, metadata, identity and errors.

    Return code semantics are encoded in ``comparison.json``: 0 means all
    checks passed, 1 means both sides ran but differed, and 2 means execution
    or evidence capture failed. The output directory must not already exist.
    """
    directory = Path(output_dir).expanduser().resolve()
    if directory.exists():
        raise FileExistsError(f"Evidence directory already exists: {directory}")
    if selection.target != "x5":
        raise ValueError("FCOS evaluator only supports explicit target x5.")
    if resize_type != 0:
        raise ValueError("FCOS source comparison only supports resize_type=0; letterbox is evaluated by task tests.")
    if not 0 <= conf_thres <= 1 or not 0 <= iou_thres <= 1:
        raise ValueError("conf_thres and iou_thres must be in [0,1].")
    if type(priority) is not int or not 0 <= priority <= 255:
        raise ValueError("priority must be between 0 and 255.")
    cores = [0] if bpu_cores is None else list(bpu_cores)
    if not cores or any(type(core) is not int or core < 0 for core in cores):
        raise ValueError("bpu_cores must contain non-negative integers.")
    if not isinstance(image, np.ndarray) or image.ndim != 3 or image.dtype != np.uint8:
        raise ValueError("image must be a uint8 BGR array.")
    directory.mkdir(parents=True, exist_ok=False)
    started = datetime.now(timezone.utc).isoformat()
    summary: dict[str, Any] = {
        "source_ref": SOURCE_REF,
        "started_utc": started,
        "argv": list(sys.argv) if command is None else list(command),
        "command": list(sys.argv) if command is None else list(command),
        "cwd": str(Path.cwd()),
        "target": selection.target,
        "asset_id": selection.asset_id,
        "variant": selection.variant,
        "model_path": str(selection.model_path.resolve()),
        "image_path": str(Path(image_path).expanduser().resolve()),
        "resize_type": resize_type,
        "conf_thres": conf_thres,
        "iou_thres": iou_thres,
        "priority": priority,
        "bpu_cores": cores,
        "host": {"python": sys.version, "numpy": np.__version__, "opencv": cv2.__version__},
        "board_identity": {},
        "hashes": {},
        "checks": {},
        "passed": False,
    }
    arrays: dict[str, Any] = {}
    source_data: dict[str, Any] | None = None
    unified_data: dict[str, Any] | None = None
    try:
        actual = require_execution_target(selection.target)
        summary["board_identity"]["resolved_target"] = actual
        for path in (Path("/sys/class/boardinfo/soc_name"), Path("/sys/class/boardinfo/board_type")):
            summary["board_identity"][str(path)] = path.read_text(encoding="utf-8").strip() if path.is_file() else None
        summary["hashes"]["model_sha256"] = _sha256(selection.model_path)
        summary["hashes"]["image_file_sha256"] = _sha256(Path(image_path).expanduser())
        np.save(directory / "input.npy", image, allow_pickle=False)
        summary["hashes"]["input_npy_sha256"] = _sha256(directory / "input.npy")
        code_files = [
            SOURCE_FILE,
            ROOT / "platforms" / "x5" / "utils" / "py_utils" / "preprocess.py",
            ROOT / "platforms" / "x5" / "utils" / "py_utils" / "postprocess.py",
            Path(__file__),
            *sorted((SAMPLE_DIR / "runtime" / "python").glob("*.py")),
            *(ROOT / "samples" / "_shared" / name for name in ("assets.py", "platforms.py", "runtime_meta.py", "quantization.py", "image.py")),
        ]
        summary["hashes"]["code_sha256"] = {
            str(path.relative_to(ROOT)): _sha256(path) for path in code_files if path.is_file()
        }
        if legacy_runner_factory is None:
            source_data = _run_source(selection, image, resize_type=resize_type, conf_thres=conf_thres, iou_thres=iou_thres, priority=priority, bpu_cores=cores)
        else:
            source_data = legacy_runner_factory(selection, image, resize_type=resize_type, conf_thres=conf_thres, iou_thres=iou_thres, priority=priority, bpu_cores=cores)
        source_data = _normalise_side(source_data)
        _save_side(directory / "source", source_data, arrays)
        if runtime_factory is None:
            from samples.vision.fcos.runtime.python.model_runner import _default_runtime_factory

            runtime_factory = _default_runtime_factory()
        unified_data = _run_unified(selection, image, resize_type=resize_type, conf_thres=conf_thres, iou_thres=iou_thres, priority=priority, bpu_cores=cores, runtime_factory=runtime_factory)
        unified_data = _normalise_side(unified_data)
        _save_side(directory / "unified", unified_data, arrays)
        summary["checks"].update(_compare_arrays(source_data["inputs"], unified_data["inputs"], "inputs"))
        summary["checks"].update(_compare_arrays(source_data["raw"], unified_data["raw"], "raw"))
        summary["checks"].update(_compare_arrays(source_data["result"], unified_data["result"], "result"))
        summary["checks"]["metadata"] = source_data["metadata"] == unified_data["metadata"]
        summary["passed"] = bool(summary["checks"]) and all(summary["checks"].values())
        summary["return_code"] = 0 if summary["passed"] else 1
    except Exception as exc:
        _record_error(summary, "execution", exc)
        summary["return_code"] = 2
    finally:
        if _FIXED_SOURCE_MODULES:
            sys.modules.update(_FIXED_SOURCE_MODULES)
        summary["arrays"] = arrays
        summary["finished_utc"] = datetime.now(timezone.utc).isoformat()
        _save_json(directory / "comparison.json", summary)
        if summary.get("errors"):
            _save_json(directory / "errors.json", summary["errors"])
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare fixed-source and unified FCOS raw/runtime evidence on X5.")
    parser.add_argument("--target", choices=("x5",), required=True)
    parser.add_argument("--asset-id", default=None)
    parser.add_argument("--variant", choices=("efficientnetb0", "efficientnetb2", "efficientnetb3"), default=None)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--test-img", default=str(SAMPLE_DIR / "test_data" / "bus.jpg"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resize-type", type=int, choices=(0,), default=0)
    parser.add_argument("--conf-thres", type=float, default=0.5)
    parser.add_argument("--iou-thres", type=float, default=0.6)
    parser.add_argument("--priority", type=int, default=0)
    parser.add_argument("--bpu-cores", type=int, nargs="+", default=[0])
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        selection = resolve_selection(args.target, asset_id=args.asset_id, variant=args.variant, model_path=args.model_path)
        image_path = Path(args.test_img).expanduser()
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"input image not found or unreadable: {image_path}")
        effective_argv = list(sys.argv) if argv is None else [sys.argv[0], *argv]
        summary = run_comparison(selection, image, image_path, args.output_dir, resize_type=args.resize_type, conf_thres=args.conf_thres, iou_thres=args.iou_thres, priority=args.priority, bpu_cores=args.bpu_cores, command=effective_argv)
        print(json.dumps({"passed": summary["passed"], "return_code": summary["return_code"], "evidence": str(Path(args.output_dir).expanduser().resolve())}, ensure_ascii=False))
        return int(summary["return_code"])
    except (OSError, RuntimeError, ValueError, TypeError, ImportError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
