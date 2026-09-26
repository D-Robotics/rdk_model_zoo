# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0

"""Manifest selection and runtime metadata binding for UNet."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from samples._shared.assets import Asset, list_assets
from samples._shared.quantization import validate_scale_quantization as _validate_quant
from samples._shared.runtime_meta import MetadataMismatchError, RuntimeMetadata


class BindingError(ValueError):
    """Base class for UNet selection and metadata errors."""


SUPPORTED_TARGETS = ("x5", "s100", "s100p", "s600")
SAMPLE_DIR = Path(__file__).resolve().parents[2]
INPUT_SIZE = 512
NUM_CLASSES = 21


@dataclass(frozen=True)
class ModelSelection:
    """One published manifest asset and its selected local path."""

    target: str
    asset: Asset
    model_path: Path
    explicit_model_path: bool = False
    variant: str = "resnet18"


@dataclass(frozen=True)
class ModelBinding:
    """Validated UNet tensor protocol; fixed 512x512 Pascal VOC geometry."""

    selection: ModelSelection
    metadata: RuntimeMetadata
    input_name: str
    output_name: str

    @property
    def model_name(self) -> str:
        return self.metadata.model_name


VARIANTS = ("resnet18", "resnet34", "resnet50", "resnet101", "resnet152")


def list_available_assets(target=None):
    if target in (None, "auto", "x5"):
        return tuple(list_assets("x5", "unet"))
    if target in SUPPORTED_TARGETS:
        return ()
    raise BindingError(f"Unknown target {target!r}")


def resolve_selection(target="auto", *, variant=None, asset_id=None, model_path=None):
    key = (target or "auto").lower()
    if key == "auto":
        key = "x5"
    if key != "x5":
        raise BindingError("UNet has published assets only for x5.")
    rows = list_available_assets(key)
    if asset_id is not None:
        matches = [a for a in rows if a.reference == asset_id]
        if len(matches) != 1:
            raise BindingError(f"Unknown UNet asset {asset_id!r}")
        inferred = matches[0].filename.removeprefix("unet_").split("_voc_")[0]
        if variant is not None and variant != inferred:
            raise BindingError("variant and asset-id select different UNet artifacts")
        variant = inferred
    variant = variant or "resnet18"
    if variant not in VARIANTS:
        raise BindingError(f"Unknown UNet backbone {variant!r}")
    filename = f"unet_{variant}_voc_512x512_nv12.bin"
    matches = [a for a in rows if a.filename == filename]
    if len(matches) != 1:
        raise BindingError(f"Expected one manifest asset for {variant}")
    if model_path is not None and asset_id is None:
        raise BindingError("External model paths require the exact --asset-id")
    return ModelSelection(key, matches[0], Path(model_path).expanduser() if model_path else SAMPLE_DIR/"model"/filename,
                          model_path is not None, variant)


def bind_model(selection: ModelSelection, metadata: RuntimeMetadata | Mapping[str, Any]) -> ModelBinding:
    """Validate the fixed source UNet tensor protocol."""

    # Re-resolve caller-created selections: identity and path stay inseparable.
    resolved = resolve_selection(
        selection.target,
        variant=selection.variant,
        asset_id=selection.asset.reference,
        model_path=selection.model_path if selection.explicit_model_path else None,
    )
    if (
        selection.target != resolved.target
        or selection.asset != resolved.asset
        or Path(selection.model_path) != Path(resolved.model_path)
        or selection.explicit_model_path != resolved.explicit_model_path
    ):
        raise BindingError("ModelSelection does not match the exact manifest asset and path.")

    meta = metadata if isinstance(metadata, RuntimeMetadata) else RuntimeMetadata.from_mapping(metadata)
    if meta.model_names != (meta.model_name,):
        raise MetadataMismatchError("UNet artifact must expose exactly one model.")
    if len(meta.input_names) != 1 or len(meta.output_names) != 1:
        raise MetadataMismatchError("UNet requires exactly one input and one output.")
    input_name, output_name = meta.input_names[0], meta.output_names[0]
    shape = meta.input_shapes.get(input_name, ())
    if shape not in ((1, 3, 512, 512), (1, 512, 512, 3), (1, 768, 512, 1)):
        raise MetadataMismatchError("UNet NV12 input metadata must describe 512x512 geometry.")
    if meta.input_dtypes.get(input_name) != "nv12":
        raise MetadataMismatchError("UNet requires an NV12 input, not a float featuremap.")
    output_shape = meta.output_shapes.get(output_name, ())
    if output_shape not in ((1, 21, 512, 512), (1, 512, 512, 21)):
        raise MetadataMismatchError("UNet output must be NCHW/NHWC logits for 21 classes at 512x512.")
    dtype = meta.output_dtypes.get(output_name)
    if dtype not in ("float32", "int8", "uint8", "int16", "int32"):
        raise MetadataMismatchError(f"Unsupported UNet output dtype {dtype!r}.")
    if dtype != "float32":
        _validate_quant(meta.output_quants.get(output_name), output_shape)
    return ModelBinding(selection, meta, input_name, output_name)
