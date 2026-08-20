#!/usr/bin/env python3
"""Export the fused HIMLoco TorchScript policy to a fixed-shape ONNX model."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import onnx
import torch
from onnx.reference import ReferenceEvaluator


INPUT_NAME = "obs_history"
OUTPUT_NAME = "actions"
INPUT_SHAPE = (1, 270)
OUTPUT_SHAPE = (1, 12)


def sha256(path: Path) -> str:
    """Return the SHA256 digest of one artifact."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fixed_shape(value_info: onnx.ValueInfoProto) -> tuple[int, ...]:
    """Read a fully static tensor shape from ONNX value information."""

    dimensions = value_info.type.tensor_type.shape.dim
    if any(not dimension.HasField("dim_value") for dimension in dimensions):
        raise ValueError(f"{value_info.name} must use a fully static shape")
    return tuple(int(dimension.dim_value) for dimension in dimensions)


def cosine_similarity(candidate: np.ndarray, reference: np.ndarray) -> float:
    """Calculate cosine similarity with deterministic zero-vector handling."""

    candidate64 = candidate.astype(np.float64).ravel()
    reference64 = reference.astype(np.float64).ravel()
    denominator = np.linalg.norm(candidate64) * np.linalg.norm(reference64)
    if denominator == 0.0:
        return 1.0 if np.array_equal(candidate64, reference64) else 0.0
    similarity = np.dot(candidate64, reference64) / denominator
    return float(np.clip(similarity, -1.0, 1.0))


def validate_export(
    jit_model: torch.jit.ScriptModule,
    onnx_model: onnx.ModelProto,
    samples: int,
    seed: int,
) -> dict[str, float | int | bool]:
    """Compare TorchScript and ONNX ReferenceEvaluator outputs."""

    evaluator = ReferenceEvaluator(onnx_model)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    inputs = [torch.zeros(INPUT_SHAPE, dtype=torch.float32)]
    inputs.extend(
        torch.randn(INPUT_SHAPE, generator=generator, dtype=torch.float32)
        for _ in range(samples - 1)
    )

    references: list[np.ndarray] = []
    candidates: list[np.ndarray] = []
    with torch.inference_mode():
        for tensor in inputs:
            reference = jit_model(tensor)
            if not isinstance(reference, torch.Tensor):
                raise TypeError("TorchScript forward must return one Tensor")
            reference_array = reference.detach().cpu().numpy().astype(np.float32)
            if reference_array.shape != OUTPUT_SHAPE:
                raise ValueError(
                    f"unexpected TorchScript output: {reference_array.shape} != {OUTPUT_SHAPE}"
                )
            candidate = evaluator.run([OUTPUT_NAME], {INPUT_NAME: tensor.numpy()})[
                0
            ].astype(np.float32)
            references.append(reference_array)
            candidates.append(candidate)

    reference_all = np.concatenate(references, axis=0).astype(np.float64)
    candidate_all = np.concatenate(candidates, axis=0).astype(np.float64)
    delta = candidate_all - reference_all
    per_sample_cosine = [
        cosine_similarity(candidate, reference)
        for candidate, reference in zip(candidates, references, strict=True)
    ]
    return {
        "sample_count": samples,
        "mae": float(np.abs(delta).mean()),
        "rmse": float(np.sqrt(np.square(delta).mean())),
        "max_abs": float(np.abs(delta).max()),
        "global_cosine_similarity": cosine_similarity(candidate_all, reference_all),
        "minimum_sample_cosine_similarity": float(min(per_sample_cosine)),
        "allclose_rtol_1e-5_atol_1e-6": bool(
            np.allclose(candidate_all, reference_all, rtol=1e-5, atol=1e-6)
        ),
    }


def main() -> None:
    """Export, validate, and report one fused HIMLoco ONNX model."""

    parser = argparse.ArgumentParser(
        description="Export fused HIMLoco policy.pt to fixed-shape ONNX."
    )
    parser.add_argument(
        "--jit", required=True, type=Path, help="Fused TorchScript policy.pt."
    )
    parser.add_argument(
        "--output", required=True, type=Path, help="Destination ONNX path."
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Export report path; defaults to <output-stem>.export.json.",
    )
    parser.add_argument("--opset", type=int, default=11, help="ONNX opset version.")
    parser.add_argument(
        "--validation-samples",
        type=int,
        default=8,
        help="Number of zero/random inputs used for numerical validation.",
    )
    parser.add_argument(
        "--seed", type=int, default=20260820, help="Validation RNG seed."
    )
    args = parser.parse_args()

    jit_path = args.jit.resolve()
    output_path = args.output.resolve()
    report_path = (
        args.report.resolve()
        if args.report is not None
        else output_path.with_suffix(".export.json")
    )
    if not jit_path.is_file():
        raise FileNotFoundError(jit_path)
    if report_path == output_path:
        raise ValueError("--report and --output must be different paths")
    if output_path.exists():
        raise FileExistsError(output_path)
    if report_path.exists():
        raise FileExistsError(report_path)
    if args.validation_samples < 1:
        raise ValueError("--validation-samples must be positive")
    if args.opset != 11:
        raise ValueError(
            "this RDK X5 conversion contract is validated only with opset 11"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    jit_model = torch.jit.load(str(jit_path), map_location="cpu").eval()
    dummy_input = torch.zeros(INPUT_SHAPE, dtype=torch.float32)
    with torch.inference_mode():
        smoke_output = jit_model(dummy_input)
    if (
        not isinstance(smoke_output, torch.Tensor)
        or tuple(smoke_output.shape) != OUTPUT_SHAPE
    ):
        shape = (
            tuple(smoke_output.shape)
            if isinstance(smoke_output, torch.Tensor)
            else type(smoke_output)
        )
        raise ValueError(f"unexpected TorchScript output contract: {shape}")

    temporary_handle = tempfile.NamedTemporaryFile(
        prefix=f".{output_path.stem}.",
        suffix=".onnx",
        dir=output_path.parent,
        delete=False,
    )
    temporary_path = Path(temporary_handle.name)
    temporary_handle.close()
    try:
        torch.onnx.export(
            jit_model,
            dummy_input,
            str(temporary_path),
            export_params=True,
            opset_version=args.opset,
            input_names=[INPUT_NAME],
            output_names=[OUTPUT_NAME],
            dynamic_axes={},
            do_constant_folding=True,
            dynamo=False,
        )
        onnx_model = onnx.load(temporary_path)
        onnx.checker.check_model(onnx_model)
        if len(onnx_model.graph.input) != 1 or len(onnx_model.graph.output) != 1:
            raise ValueError("ONNX must expose exactly one input and one output")
        if onnx_model.graph.input[0].name != INPUT_NAME:
            raise ValueError(
                f"unexpected ONNX input name: {onnx_model.graph.input[0].name}"
            )
        if onnx_model.graph.output[0].name != OUTPUT_NAME:
            raise ValueError(
                f"unexpected ONNX output name: {onnx_model.graph.output[0].name}"
            )
        if (
            onnx_model.graph.input[0].type.tensor_type.elem_type
            != onnx.TensorProto.FLOAT
        ):
            raise ValueError("ONNX input dtype is not float32")
        if (
            onnx_model.graph.output[0].type.tensor_type.elem_type
            != onnx.TensorProto.FLOAT
        ):
            raise ValueError("ONNX output dtype is not float32")
        if fixed_shape(onnx_model.graph.input[0]) != INPUT_SHAPE:
            raise ValueError("ONNX input shape is not [1, 270]")
        if fixed_shape(onnx_model.graph.output[0]) != OUTPUT_SHAPE:
            raise ValueError("ONNX output shape is not [1, 12]")

        metrics = validate_export(
            jit_model, onnx_model, args.validation_samples, args.seed
        )
        if not metrics["allclose_rtol_1e-5_atol_1e-6"]:
            raise ValueError(f"TorchScript/ONNX validation failed: {metrics}")
        os.replace(temporary_path, output_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise

    report = {
        "schema_version": "1.0",
        "source": str(jit_path),
        "source_sha256": sha256(jit_path),
        "output": str(output_path),
        "output_sha256": sha256(output_path),
        "torch_version": torch.__version__,
        "onnx_version": onnx.__version__,
        "opset": args.opset,
        "input_contract": {
            "name": INPUT_NAME,
            "shape": list(INPUT_SHAPE),
            "dtype": "float32",
            "history": "current observation followed by five previous 45-value observations",
        },
        "output_contract": {
            "name": OUTPUT_NAME,
            "shape": list(OUTPUT_SHAPE),
            "dtype": "float32",
        },
        "operators": sorted({node.op_type for node in onnx_model.graph.node}),
        "validation_seed": args.seed,
        "validation": metrics,
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
