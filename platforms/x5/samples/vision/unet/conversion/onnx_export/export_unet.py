#!/usr/bin/env python3
# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0

"""Export a fixed-shape UNet ResNet checkpoint to an X5 PTQ-ready ONNX."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from model import RESNET_SPECS, UNet

INPUT_NAME = "images"
OUTPUT_NAME = "logits"
INPUT_SHAPE = (1, 3, 512, 512)
OUTPUT_SHAPE = (1, 21, 512, 512)
ONNX_OPSET = 11
ORT_RTOL = 1e-4
ORT_ATOL = 5e-5


def sha256_file(path: Path) -> str:
    """Return the lowercase SHA256 digest of one file."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_state_dict(checkpoint: object) -> dict[str, Any]:
    """Extract and normalize a model state dictionary from a checkpoint."""

    if not isinstance(checkpoint, dict):
        raise TypeError("checkpoint must contain a state dictionary")
    if "model_state_dict" in checkpoint:
        state = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint:
        state = checkpoint["state_dict"]
    else:
        state = checkpoint
    if not isinstance(state, dict) or not state:
        raise TypeError("checkpoint state dictionary is empty or invalid")
    if all(isinstance(name, str) and name.startswith("module.") for name in state):
        return {name.removeprefix("module."): value for name, value in state.items()}
    return state


def load_checkpoint(path: Path) -> object:
    """Load weights safely when supported while retaining torch 1.x support."""

    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def build_model(backbone: str, checkpoint_path: Path) -> UNet:
    """Build one UNet variant and strictly load its checkpoint."""

    model = UNet(backbone=backbone)
    state = extract_state_dict(load_checkpoint(checkpoint_path))
    result = model.load_state_dict(state, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(f"strict checkpoint load failed: {result}")
    return model.eval()


def static_shape(value_info: Any) -> tuple[int, ...]:
    """Read a concrete ONNX tensor shape and reject dynamic dimensions."""

    dimensions: list[int] = []
    for dimension in value_info.type.tensor_type.shape.dim:
        if not dimension.HasField("dim_value") or dimension.dim_value <= 0:
            raise ValueError(f"dynamic ONNX shape is not supported: {value_info.name}")
        dimensions.append(int(dimension.dim_value))
    return tuple(dimensions)


def check_onnx_contract(path: Path) -> dict[str, object]:
    """Run ONNX checker and verify the fixed UNet input/output contract."""

    try:
        import onnx
    except ImportError as exc:
        raise RuntimeError("onnx is required to validate the exported model") from exc

    graph = onnx.load(str(path))
    onnx.checker.check_model(graph)
    initializer_names = {initializer.name for initializer in graph.graph.initializer}
    inputs = [item for item in graph.graph.input if item.name not in initializer_names]
    outputs = list(graph.graph.output)
    if len(inputs) != 1 or len(outputs) != 1:
        raise ValueError("UNet ONNX must contain exactly one input and one output")
    if inputs[0].name != INPUT_NAME or outputs[0].name != OUTPUT_NAME:
        raise ValueError(
            "unexpected ONNX tensor names: "
            f"{inputs[0].name!r} -> {outputs[0].name!r}"
        )
    if static_shape(inputs[0]) != INPUT_SHAPE:
        raise ValueError(f"unexpected ONNX input shape: {static_shape(inputs[0])}")
    if static_shape(outputs[0]) != OUTPUT_SHAPE:
        raise ValueError(f"unexpected ONNX output shape: {static_shape(outputs[0])}")
    opsets = {
        item.domain or "ai.onnx": int(item.version) for item in graph.opset_import
    }
    if opsets.get("ai.onnx") != ONNX_OPSET:
        raise ValueError(f"expected ONNX opset {ONNX_OPSET}, got {opsets}")
    return {
        "onnx_version": onnx.__version__,
        "opsets": opsets,
        "input_name": inputs[0].name,
        "input_shape": list(INPUT_SHAPE),
        "output_name": outputs[0].name,
        "output_shape": list(OUTPUT_SHAPE),
    }


def compare_onnx_runtime(
    path: Path,
    sample: Any,
    reference: np.ndarray,
) -> dict[str, object]:
    """Compare ONNX Runtime output with the PyTorch reference output."""

    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError(
            "onnxruntime is required for the default numerical export check; "
            "pass --skip-runtime-check only for a structural preflight"
        ) from exc

    session = ort.InferenceSession(
        str(path),
        providers=["CPUExecutionProvider"],
    )
    candidate = np.asarray(session.run([OUTPUT_NAME], {INPUT_NAME: sample.numpy()})[0])
    if tuple(candidate.shape) != OUTPUT_SHAPE:
        raise ValueError(f"unexpected ONNX Runtime output shape: {candidate.shape}")
    absolute = np.abs(candidate.astype(np.float64) - reference.astype(np.float64))
    relative = absolute / np.maximum(np.abs(reference.astype(np.float64)), 1e-8)
    np.testing.assert_allclose(
        candidate,
        reference,
        rtol=ORT_RTOL,
        atol=ORT_ATOL,
    )
    return {
        "status": "passed",
        "runtime_version": ort.__version__,
        "max_absolute_error": float(absolute.max(initial=0.0)),
        "max_relative_error": float(relative.max(initial=0.0)),
        "rtol": ORT_RTOL,
        "atol": ORT_ATOL,
    }


def export_model(
    backbone: str,
    checkpoint_path: Path,
    output_path: Path,
    report_path: Path,
    skip_runtime_check: bool,
) -> dict[str, object]:
    """Export, validate, atomically publish, and describe one ONNX model."""

    import torch

    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite ONNX model: {output_path}")
    if report_path.exists():
        raise FileExistsError(f"refusing to overwrite export report: {report_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    model = build_model(backbone, checkpoint_path)
    torch.manual_seed(11)
    sample = torch.rand(INPUT_SHAPE, dtype=torch.float32)
    with torch.inference_mode():
        reference = model(sample).detach().cpu().numpy()
    if tuple(reference.shape) != OUTPUT_SHAPE:
        raise ValueError(f"unexpected PyTorch output shape: {reference.shape}")

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=output_path.parent,
            prefix=f".{output_path.stem}.",
            suffix=".onnx",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
        torch.onnx.export(
            model,
            sample,
            str(temporary_path),
            input_names=[INPUT_NAME],
            output_names=[OUTPUT_NAME],
            opset_version=ONNX_OPSET,
            do_constant_folding=True,
        )
        contract = check_onnx_contract(temporary_path)
        numerical = (
            {"status": "skipped", "reason": "--skip-runtime-check"}
            if skip_runtime_check
            else compare_onnx_runtime(temporary_path, sample, reference)
        )
        os.chmod(temporary_path, 0o644)
        os.replace(temporary_path, output_path)
        temporary_path = None
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()

    report = {
        "schema_version": "1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_family": "unet_resnet",
        "backbone": backbone,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "onnx": str(output_path),
        "onnx_sha256": sha256_file(output_path),
        "torch_version": torch.__version__,
        "contract": contract,
        "numerical_check": numerical,
        "x5_ptq_ready": numerical["status"] == "passed",
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", choices=tuple(RESNET_SPECS), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--report",
        type=Path,
        help="new JSON report path; defaults to <output-stem>.export.json",
    )
    parser.add_argument(
        "--skip-runtime-check",
        action="store_true",
        help="skip PyTorch/ONNX Runtime comparison and mark PTQ readiness false",
    )
    return parser.parse_args()


def main() -> int:
    """Run the guarded ONNX export."""

    args = parse_args()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    report_path = (
        args.report.expanduser().resolve()
        if args.report
        else output_path.with_suffix(".export.json")
    )
    report = export_model(
        backbone=args.backbone,
        checkpoint_path=checkpoint_path,
        output_path=output_path,
        report_path=report_path,
        skip_runtime_check=args.skip_runtime_check,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
