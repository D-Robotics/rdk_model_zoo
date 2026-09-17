#!/usr/bin/env python3
"""Export the TorchVision ResNet18 classification graph to ONNX.

The original X5 and S18 samples document TorchVision ResNet18 as their source
model but do not ship an export script. This small, explicit exporter fills that
source step without embedding any board conversion commands. Imports for the
optional ML stack are delayed until an export is requested, so ``--help`` is
usable on a clean host.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence


def build_parser() -> argparse.ArgumentParser:
    """Build the exporter CLI without importing PyTorch or TorchVision."""

    parser = argparse.ArgumentParser(
        description="Export TorchVision ResNet18 to a fixed 224x224 ONNX graph."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("resnet18.onnx"),
        help="Output ONNX path (default: ./resnet18.onnx).",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=11,
        help="ONNX opset passed to torch.onnx.export (default: 11).",
    )
    parser.add_argument(
        "--weights",
        choices=("IMAGENET1K_V1", "none"),
        default="IMAGENET1K_V1",
        help=(
            "TorchVision weight source (default: explicit IMAGENET1K_V1; "
            "downloads official weights when absent)."
        ),
    )
    parser.add_argument(
        "--no-check",
        action="store_true",
        help="Skip onnx.checker after export (use only when ONNX is unavailable).",
    )
    return parser


def export_resnet18(
    output: str | Path,
    *,
    opset: int = 11,
    weights: str = "IMAGENET1K_V1",
    check: bool = True,
) -> Path:
    """Export one fixed-shape ResNet18 graph and return its path.

    The generated graph has one NCHW input named ``data`` with shape
    ``[1, 3, 224, 224]`` and one score output named ``output`` with shape
    ``[1, 1000]``. Runtime NV12 conversion remains the board conversion
    contract and is handled by the selected OE configuration.
    """

    if opset < 11:
        raise ValueError("ResNet18 export requires ONNX opset 11 or newer.")
    destination = Path(output).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)

    try:
        import torch
        from torchvision.models import ResNet18_Weights, resnet18
    except ImportError as exc:
        raise RuntimeError(
            "Export requires PyTorch and TorchVision; install them in the "
            "conversion environment described by conversion/README.md."
        ) from exc

    if weights not in ("IMAGENET1K_V1", "none"):
        raise ValueError("weights must be IMAGENET1K_V1 or none.")
    model_weights = (
        ResNet18_Weights.IMAGENET1K_V1 if weights == "IMAGENET1K_V1" else None
    )
    model = resnet18(weights=model_weights)
    model.eval()
    dummy = torch.zeros((1, 3, 224, 224), dtype=torch.float32)
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy,
            str(destination),
            opset_version=opset,
            input_names=["data"],
            output_names=["output"],
            dynamic_axes=None,
            do_constant_folding=True,
            # Keep the legacy TorchScript exporter path used by the OE sample;
            # the newer dynamo exporter can alter graph structure and has not
            # been established equivalent to the published artifacts.
            dynamo=False,
        )

    if check:
        try:
            import onnx
        except ImportError as exc:
            raise RuntimeError(
                "ONNX export finished but validation needs the `onnx` package; "
                "rerun with --no-check only if validation is handled separately."
            ) from exc
        onnx.checker.check_model(onnx.load(str(destination)))
    print(f"Exported TorchVision ResNet18 ONNX graph to {destination}")
    return destination


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the exporter and return a shell status."""

    args = build_parser().parse_args(argv)
    try:
        export_resnet18(
            args.output,
            opset=args.opset,
            weights=args.weights,
            check=not args.no_check,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
