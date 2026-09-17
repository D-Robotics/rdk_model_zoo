#!/usr/bin/env python3
# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Prepare and optionally compile the YOLOE-26 PF ONNX exports for Nash S-series."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Any

import cv2
import numpy as np
import onnx
import yaml

SAMPLE = Path(__file__).resolve().parents[1]
RUNTIME = SAMPLE / "runtime" / "python"
sys.path.insert(0, str(RUNTIME))
from yoloe26seg import (  # noqa: E402
    CLASSES,
    MARCHES,
    OUTPUT_NAMES,
    SIZES,
    hbm_name,
    model_stem,
    prepare_rgb,
    read_metadata,
    sha256,
    validate_shapes,
)

IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png"}
SAMPLE_COUNT = 100
JOBS = 4
OPTIMIZE_LEVEL = "O2"


def _size_from_name(path: Path) -> str | None:
    """Extract a model size from a canonical PF artifact filename.

    Args:
        path: ONNX or metadata path whose name may contain the model stem.

    Returns:
        The size letter when present, otherwise ``None``.
    """
    match = re.search(r"yoloe_26([nsmlx])_seg_pf", path.name)
    return match.group(1) if match else None


def _find_source(root: Path, size: str, suffix: str) -> Path:
    """Resolve one source file while avoiding ambiguous experiment artifacts.

    Args:
        root: A direct artifact path or directory to search.
        size: Requested model size letter.
        suffix: Required artifact suffix, such as ``.onnx`` or ``.json``.

    Returns:
        The single source path matching the canonical model stem.
    """

    stem = model_stem(size)
    if root.is_file():
        return root
    direct = (root / size / f"{stem}{suffix}", root / f"{stem}{suffix}")
    for candidate in direct:
        if candidate.is_file():
            return candidate
    candidates = sorted(root.rglob(f"{stem}{suffix}"))
    if len(candidates) != 1:
        detail = ", ".join(str(path) for path in candidates[:8])
        raise FileNotFoundError(
            f"Expected one {stem}{suffix} below {root}, found {len(candidates)}: {detail}"
        )
    return candidates[0]


def _resolve_pairs(onnx_arg: Path, metadata_arg: Path, requested: tuple[str, ...]) -> list[dict[str, Any]]:
    """Pair ONNX and metadata sources for each requested model size.

    Args:
        onnx_arg: ONNX file or directory containing size-specific exports.
        metadata_arg: JSON file or directory containing matching metadata.
        requested: Ordered model sizes to resolve.

    Returns:
        Source dictionaries containing size, ONNX path, and metadata path.
    """
    if onnx_arg.is_file() or metadata_arg.is_file():
        inferred = _size_from_name(onnx_arg) or _size_from_name(metadata_arg)
        if inferred is None:
            raise ValueError("A single --onnx/--metadata file must use the yoloe_26<SIZE>_seg_pf stem")
        if requested != (inferred,):
            raise ValueError(f"Single-file inputs require --sizes {inferred}")

    pairs = []
    for size in requested:
        onnx_path = _find_source(onnx_arg, size, ".onnx")
        metadata_path = _find_source(metadata_arg, size, ".json")
        if _size_from_name(onnx_path) != size or _size_from_name(metadata_path) != size:
            raise ValueError(f"Source identity mismatch for size {size}: {onnx_path}, {metadata_path}")
        pairs.append({"size": size, "onnx": onnx_path.resolve(), "metadata": metadata_path.resolve()})
    return pairs


def _validate_onnx(path: Path) -> None:
    """Validate one ONNX graph against the fixed raw PF input/output contract.

    Args:
        path: ONNX graph to load and inspect.

    Returns:
        None.
    """
    model = onnx.load(str(path), load_external_data=False)
    onnx.checker.check_model(model)
    if len(model.graph.input) != 1:
        raise ValueError(f"Expected one ONNX input in {path}")
    input_value = model.graph.input[0]
    if input_value.name != "images":
        raise ValueError(f"Expected ONNX input named images, got {input_value.name!r}")
    dims = [dim.dim_value for dim in input_value.type.tensor_type.shape.dim]
    if dims != [1, 3, 640, 640]:
        raise ValueError(f"Expected input shape [1, 3, 640, 640], got {dims}")
    output_names = [value.name for value in model.graph.output]
    if output_names != list(OUTPUT_NAMES):
        raise ValueError(f"Unexpected ONNX output names: {output_names}")
    output_shapes = []
    for value in model.graph.output:
        dims = [dim.dim_value for dim in value.type.tensor_type.shape.dim]
        if any(dim <= 0 for dim in dims):
            raise ValueError(f"ONNX output {value.name} is not static: {dims}")
        output_shapes.append(dims)
    validate_shapes(output_shapes)


def _validate_source(pair: dict[str, Any]) -> dict[str, Any]:
    """Validate one ONNX/metadata/vocabulary source set.

    Args:
        pair: Source dictionary produced by ``_resolve_pairs``.

    Returns:
        The source dictionary enriched with metadata, hash, and names path.
    """
    size = pair["size"]
    onnx_path = pair["onnx"]
    metadata_path = pair["metadata"]
    if not onnx_path.is_file():
        raise FileNotFoundError(onnx_path)
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)
    metadata = read_metadata(metadata_path)
    if metadata["size"] != size:
        raise ValueError(f"Metadata size mismatch for {size}: {metadata['size']}")
    digest = sha256(onnx_path)
    if metadata.get("onnx_sha256") != digest:
        raise ValueError(f"ONNX hash mismatch for {size}: metadata={metadata.get('onnx_sha256')} actual={digest}")
    _validate_onnx(onnx_path)
    names_path = metadata_path.with_suffix(".names")
    if not names_path.is_file():
        raise FileNotFoundError(f"Missing checkpoint-ordered labels: {names_path}")
    names = names_path.read_text(encoding="utf-8").splitlines()
    if names != metadata["names"] or len(names) != CLASSES:
        raise ValueError(f"Label vocabulary mismatch for {size}: {names_path}")
    pair["metadata_data"] = metadata
    pair["onnx_sha256"] = digest
    pair["names"] = names_path
    return pair


def _image_paths(root: Path) -> list[Path]:
    """List supported calibration images in deterministic path order.

    Args:
        root: Calibration image directory to scan recursively.

    Returns:
        Sorted paths with supported image suffixes.
    """
    if not root.is_dir():
        raise FileNotFoundError(f"Calibration image directory does not exist: {root}")
    paths = sorted(
        (path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES),
        key=lambda path: str(path).lower(),
    )
    if not paths:
        raise ValueError(f"No calibration images found below {root}")
    return paths


def _prepare_calibration(images: Path, destination: Path, count: int) -> list[dict[str, Any]]:
    """Write evenly sampled normalized RGB calibration tensors.

    Args:
        images: Directory containing representative source images.
        destination: New directory for generated ``.npy`` tensors.
        count: Maximum number of images to select.

    Returns:
        Manifest records describing each source and generated tensor.
    """
    if count < 1:
        raise ValueError("--sample-count must be positive")
    if destination.exists():
        raise FileExistsError(f"Refusing to reuse calibration directory: {destination}")
    paths = _image_paths(images)
    selected = np.linspace(0, len(paths) - 1, min(count, len(paths)), dtype=int)
    destination.mkdir(parents=True)
    manifest = []
    for index, source_index in enumerate(selected):
        source = paths[int(source_index)]
        image = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Unreadable calibration image: {source}")
        tensor = prepare_rgb(image)
        target = destination / f"{index:06d}.npy"
        np.save(target, tensor)
        manifest.append({
            "source": str(source.resolve()),
            "source_sha256": sha256(source),
            "tensor": target.name,
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
        })
    return manifest


def _config(onnx_path: Path, calibration: Path, workspace: Path, size: str, march: str) -> dict[str, Any]:
    """Build an OpenExplorer configuration for one size and target march.

    Args:
        onnx_path: Validated source ONNX path.
        calibration: Directory containing prepared calibration tensors.
        workspace: Compiler working directory.
        size: Model size letter.
        march: Target architecture, ``nash-e`` or ``nash-m``.

    Returns:
        Configuration mapping ready to serialize as YAML.
    """
    return {
        "model_parameters": {
            "onnx_model": str(onnx_path),
            "march": march,
            "working_dir": str(workspace),
            "remove_node_type": "Quantize;Dequantize",
            "output_model_file_prefix": Path(hbm_name(size, march)).stem,
        },
        "input_parameters": {
            "input_name": "images",
            "input_type_rt": "nv12",
            "input_type_train": "rgb",
            "input_layout_train": "NCHW",
            "input_shape": "1x3x640x640",
            "scale_value": 1 / 255,
        },
        "calibration_parameters": {
            "cal_data_dir": str(calibration),
            "cal_data_type": "float32",
            "calibration_type": "kl",
            "quant_config": {"model_config": {"all_node_type": "int8"}},
        },
        "compiler_parameters": {
            "extra_params": {"input_no_padding": True, "output_no_padding": False},
            "jobs": JOBS,
            "compile_mode": "latency",
            "core_num": 1,
            "debug": True,
            "optimize_level": OPTIMIZE_LEVEL,
        },
    }


def _target_metadata(source: dict[str, Any], pair: dict[str, Any], march: str) -> dict[str, Any]:
    """Derive pre-compilation target metadata from validated export metadata.

    Args:
        source: Validated ONNX export metadata.
        pair: Source dictionary containing model identity and ONNX hash.
        march: Target architecture, ``nash-e`` or ``nash-m``.

    Returns:
        Target metadata marked as not yet compiled or board validated.
    """
    metadata = deepcopy(source)
    metadata.update({
        "march": march,
        "platform": "s100p" if march == "nash-m" else "s100",
        "hbm_filename": hbm_name(pair["size"], march),
        "source_onnx": str(pair["onnx"]),
        "source_onnx_sha256": pair["onnx_sha256"],
        "quantization": {"activation": "int8", "calibration": "kl", "weights": "int8"},
        "compiler_config": {
            "remove_node_type": "Quantize;Dequantize",
            "input_no_padding": True,
            "output_no_padding": False,
            "jobs": JOBS,
            "optimize_level": OPTIMIZE_LEVEL,
        },
    })
    for key in ("hbm_sha256", "hbm_bytes", "hbm_output_quantized", "hbm_output_names", "hbm_output_dtypes"):
        metadata.pop(key, None)
    metadata.setdefault("validation", {})
    metadata["validation"].update({"hbm": "not_run", "board": "not_run"})
    return metadata


def _write_json(path: Path, value: Any) -> None:
    """Write UTF-8 JSON with stable indentation and a trailing newline."""
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _compile_one(folder: Path, size: str, march: str, compiler: str, metadata: dict[str, Any]) -> dict[str, Any]:
    """Compile one prepared model and capture a non-raising result record.

    Args:
        folder: Prepared size-specific output directory.
        size: Model size letter.
        march: Target architecture.
        compiler: ``hb_compile`` executable or path.
        metadata: Target metadata to update after successful compilation.

    Returns:
        Result record containing status and output details or an error message.
    """
    config_path = folder / "config.yaml"
    workspace = folder / "bpu_output"
    expected = workspace / hbm_name(size, march)
    result: dict[str, Any] = {
        "size": size,
        "march": march,
        "status": "failed",
        "command": [compiler, "-c", str(config_path)],
    }
    log_path = folder / "compile.log"
    try:
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.run(
                [compiler, "-c", str(config_path)],
                cwd=folder,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
            )
        result["returncode"] = process.returncode
        if process.returncode != 0:
            result["error"] = f"hb_compile exited with status {process.returncode}"
            return result
        if not expected.is_file() or expected.stat().st_size == 0:
            raise FileNotFoundError(f"Compiler did not produce {expected}")
        public_hbm = folder / expected.name
        shutil.copy2(expected, public_hbm)
        digest = sha256(public_hbm)
        metadata.update({
            "hbm_sha256": digest,
            "hbm_bytes": public_hbm.stat().st_size,
            "hbm_output_quantized": True,
        })
        metadata["validation"]["hbm"] = "compiled_not_board_validated"
        _write_json(folder / f"{model_stem(size)}.json", metadata)
        result.update({
            "status": "compiled_not_board_validated",
            "hbm": str(public_hbm),
            "hbm_sha256": digest,
            "hbm_bytes": public_hbm.stat().st_size,
        })
        return result
    except Exception as error:  # Keep five-model runs independent.
        result["error"] = str(error)
        return result


def _write_manifest(root: Path, sizes: tuple[str, ...], march: str, results: dict[str, Any]) -> None:
    """Write hashes, sizes, compiler results, and contract data to a manifest.

    Args:
        root: Conversion output root.
        sizes: Ordered model sizes included in the run.
        march: Target architecture.
        results: Per-size configuration or compilation results.

    Returns:
        None.
    """
    files: dict[str, dict[str, Any]] = {}
    for size in sizes:
        folder = root / size
        entries = {}
        for path in sorted(folder.iterdir()):
            if path.is_file() and path.suffix in {".hbm", ".json", ".names"}:
                entries[path.name] = {"sha256": sha256(path), "bytes": path.stat().st_size}
        files[size] = entries
    _write_json(root / "manifest.json", {
        "protocol": "yoloe26-pf-raw-v1",
        "target_march": march,
        "sizes": list(sizes),
        "input": {"shape": [1, 3, 640, 640], "runtime": "nv12", "preprocess": "RGB letterbox 114 / 255"},
        "quantization": {"activation": "int8", "calibration": "kl", "weights": "int8"},
        "files": files,
        "results": results,
    })


def main() -> int:
    """Prepare conversion artifacts and return nonzero if compilation fails."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", required=True, type=Path,
                        help="One export directory or one yoloe_26<SIZE>_seg_pf.onnx file")
    parser.add_argument("--metadata", required=True, type=Path,
                        help="Matching export directory or one metadata JSON file")
    parser.add_argument("--cal-images", required=True, type=Path,
                        help="Representative RGB/BGR image directory")
    parser.add_argument("--march", required=True, choices=MARCHES,
                        help="nash-e for S100 or nash-m for S100P")
    parser.add_argument("--output-dir", required=True, type=Path,
                        help="New output directory; existing directories are refused")
    parser.add_argument("--sizes", nargs="+", choices=SIZES,
                        help="Model sizes to process; defaults to all five for directory inputs")
    parser.add_argument("--sample-count", type=int, default=SAMPLE_COUNT)
    parser.add_argument("--compile", action="store_true",
                        help="Immediately run hb_compile after generating configs (opt-in)")
    args = parser.parse_args()

    onnx_arg = args.onnx.resolve()
    metadata_arg = args.metadata.resolve()
    output_root = args.output_dir.resolve()
    if output_root.exists():
        parser.error(f"Refusing to reuse existing output directory: {output_root}")
    if args.sample_count < 1:
        parser.error("--sample-count must be positive")

    if args.sizes:
        sizes = tuple(dict.fromkeys(args.sizes))
    elif onnx_arg.is_file() or metadata_arg.is_file():
        inferred = _size_from_name(onnx_arg) or _size_from_name(metadata_arg)
        if inferred is None:
            parser.error("Cannot infer model size from single-file input; pass --sizes")
        sizes = (inferred,)
    else:
        sizes = SIZES

    if args.compile and shutil.which("hb_compile") is None:
        parser.error("hb_compile is unavailable; enter the OE 3.7 Docker image or configure it in PATH")

    pairs = _resolve_pairs(onnx_arg, metadata_arg, sizes)
    for pair in pairs:
        _validate_source(pair)
    calibration_root = output_root / "calibration"
    output_root.mkdir(parents=True)
    calibration_manifest = _prepare_calibration(args.cal_images.resolve(), calibration_root, args.sample_count)
    _write_json(output_root / "calibration.json", calibration_manifest)

    results: dict[str, Any] = {}
    for pair in pairs:
        size = pair["size"]
        folder = output_root / size
        folder.mkdir()
        workspace = folder / "bpu_output"
        config = _config(pair["onnx"], calibration_root, workspace, size, args.march)
        (folder / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        metadata = _target_metadata(pair["metadata_data"], pair, args.march)
        _write_json(folder / f"{model_stem(size)}.json", metadata)
        shutil.copy2(pair["names"], folder / f"{model_stem(size)}.names")
        results[size] = {"size": size, "march": args.march, "status": "config_only"}
        if args.compile:
            results[size] = _compile_one(folder, size, args.march, shutil.which("hb_compile") or "hb_compile", metadata)
        _write_json(folder / "compile_result.json", results[size])
        _write_json(output_root / "compile_results.json", results)

    _write_manifest(output_root, sizes, args.march, results)
    _write_json(output_root / "compile_results.json", results)
    print(f"Prepared {len(sizes)} YOLOE-26 PF {args.march} config(s) in {output_root}")
    if args.compile:
        print(json.dumps(results, indent=2))
    return int(args.compile and any(item["status"] != "compiled_not_board_validated" for item in results.values()))


if __name__ == "__main__":
    raise SystemExit(main())
