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
    weight_source = parser.add_mutually_exclusive_group()
    weight_source.add_argument(
        "--weights",
        choices=("IMAGENET1K_V1", "none"),
        default=None,
        help=(
            "TorchVision weight source. Omitting both weight options "
            "keeps the official default (explicit IMAGENET1K_V1; "
            "downloads official weights when absent)."
        ),
    )
    weight_source.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help=(
            "Path to a self-trained TorchVision ResNet18 state_dict "
            "(.pth, CPU-mapped). Replaces --weights; the classifier head "
            "is rebuilt for --num-classes and the checkpoint must match "
            "that architecture exactly (strict load)."
        ),
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=None,
        help=(
            "Class count of the self-trained checkpoint (required with "
            "--checkpoint; >= 2). Determines the ONNX output width and "
            "the runtime custom_selection class_count."
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
    checkpoint: str | Path | None = None,
    num_classes: int | None = None,
    check: bool = True,
) -> Path:
    """Export one fixed-shape ResNet18 graph and return its path.

    The generated graph has one NCHW input named ``data`` with shape
    ``[1, 3, 224, 224]`` float32 and one score output named ``output``
    whose width is ``1000`` for official weights or ``num_classes`` for a
    self-trained ``checkpoint`` (the classifier head is rebuilt for that
    count and the checkpoint must load strictly against it). Runtime NV12
    conversion remains the board conversion contract and is handled by
    the selected OE configuration.
    """

    if opset < 11:
        raise ValueError("ResNet18 export requires ONNX opset 11 or newer.")
    destination = Path(output).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)

    if checkpoint is not None:
        if not isinstance(num_classes, int) or isinstance(num_classes, bool) \
                or num_classes < 2:
            raise ValueError(
                "--num-classes must be an integer >= 2 for a self-trained "
                "checkpoint (the classifier head needs at least two classes).")
        checkpoint_path = Path(checkpoint).expanduser()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"self-trained checkpoint not found: {checkpoint_path}")
        # A checkpoint is authoritative: official weights are not consulted.
        # The CLI's mutually exclusive group keeps --checkpoint from being
        # combined with --weights at all.

    try:
        import torch
        import torchvision
        from torchvision.models import ResNet18_Weights, resnet18
    except ImportError as exc:
        raise RuntimeError(
            "Export requires PyTorch and TorchVision; install them in the "
            "conversion environment described by conversion/README.md."
        ) from exc

    if checkpoint is not None:
        model = resnet18(weights=None)
        # Rebuild the classifier head for the declared class count, then
        # require the checkpoint to match the modified architecture
        # exactly: a strict load proves the fc width and every parameter
        # agree instead of silently keeping random weights.
        model.fc = torch.nn.Linear(model.fc.in_features, int(num_classes))
        state = torch.load(str(checkpoint_path), map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state, strict=True)
        class_count = int(num_classes)
    else:
        if weights not in ("IMAGENET1K_V1", "none"):
            raise ValueError("weights must be IMAGENET1K_V1 or none.")
        model_weights = (
            ResNet18_Weights.IMAGENET1K_V1 if weights == "IMAGENET1K_V1" else None
        )
        model = resnet18(weights=model_weights)
        class_count = 1000
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
    print(f"  output width: {class_count} classes")
    print(f"  training stack: torch {torch.__version__}, "
          f"torchvision {torchvision.__version__}")
    if checkpoint is not None:
        print(
            "  runtime contract: custom_selection(..., class_count="
            f"{class_count}) with the compiled artifact; record this stack "
            "in your training provenance.")
    return destination


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the exporter and return a shell status."""

    args = build_parser().parse_args(argv)
    if args.checkpoint is not None and args.num_classes is None:
        print("error: --checkpoint requires --num-classes")
        return 2
    if args.checkpoint is None and args.num_classes is not None:
        print("error: --num-classes only describes a --checkpoint export; "
              "official weights always output 1000 classes")
        return 2
    try:
        export_resnet18(
            args.output,
            opset=args.opset,
            # Neither weight option given keeps the official default; a
            # checkpoint makes weights irrelevant (argparse already
            # rejected combining the two).
            weights=args.weights or "IMAGENET1K_V1",
            checkpoint=args.checkpoint,
            num_classes=args.num_classes,
            check=not args.no_check,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
