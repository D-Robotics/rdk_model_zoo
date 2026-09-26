# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Exact asset selection and four-input/four-output planning binding."""

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from samples._shared.assets import list_assets
from samples._shared.platforms import resolve_target
from samples._shared.runtime_meta import RuntimeMetadata, MetadataMismatchError
from samples.vision.diffusiondrive.runtime.python.quantization import transform

SAMPLE_DIR = Path(__file__).resolve().parents[2]
TARGETS = ("s100p", "s600")
INPUT_SHAPES = MappingProxyType(
    {
        "camera": (1, 3, 256, 1024),
        "lidar": (1, 1, 256, 256),
        "status": (1, 8),
        "noise": (1, 20, 8, 2),
    }
)
OUTPUT_SHAPES = MappingProxyType(
    {
        "trajectory": (1, 8, 3),
        "agent_states": (1, 30, 5),
        "agent_labels": (1, 30),
        "bev_semantic_map": (1, 7, 128, 256),
    }
)


@dataclass(frozen=True)
class ModelSelection:
    target: str
    asset: object
    model_path: Path
    explicit_model_path: bool = False


@dataclass(frozen=True)
class ModelBinding:
    selection: ModelSelection
    metadata: RuntimeMetadata
    input_transforms: object
    output_transforms: object

    @property
    def model_name(self):
        return self.metadata.model_name


def list_available_assets(target=None):
    rows = tuple(list_assets("s", "diffusiondrive"))
    if target in (None, "auto"):
        return rows
    if target not in ("x5", "s100", "s100p", "s600"):
        raise ValueError(f"Unknown target {target}")
    return tuple(a for a in rows if a.filename.startswith(target + "/"))


def resolve_selection(target="auto", *, asset_id=None, model_path=None):
    rows = list_available_assets()
    if asset_id is not None:
        rows = tuple(a for a in rows if a.reference == asset_id)
        if len(rows) != 1:
            raise ValueError("Unknown DiffusionDrive asset identity")
        asset_target = rows[0].filename.split("/")[0]
        if target == "auto":
            target = asset_target
    selected = resolve_target(target)
    if selected not in TARGETS:
        raise ValueError("DiffusionDrive assets support S100P and S600 only")
    rows = tuple(a for a in rows if a.filename.startswith(selected + "/"))
    if len(rows) != 1:
        raise ValueError("Target and asset identity must match exactly")
    if model_path is not None and asset_id is None:
        raise ValueError("External model path requires exact --asset-id")
    asset = rows[0]
    return ModelSelection(
        selected,
        asset,
        (
            Path(model_path).expanduser()
            if model_path is not None
            else SAMPLE_DIR / "model" / asset.filename
        ),
        model_path is not None,
    )


def bind_model(selection, metadata):
    expected = resolve_selection(
        selection.target,
        asset_id=selection.asset.reference,
        model_path=selection.model_path if selection.explicit_model_path else None,
    )
    if selection != expected:
        raise ValueError("Selection differs from manifest contract")
    meta = (
        metadata
        if isinstance(metadata, RuntimeMetadata)
        else RuntimeMetadata.from_mapping(metadata)
    )
    if len(meta.model_names) != 1:
        raise MetadataMismatchError("Expected one planning model")
    transforms = []
    for direction, shapes in (("input", INPUT_SHAPES), ("output", OUTPUT_SHAPES)):
        names = getattr(meta, direction + "_names")
        actual_shapes = getattr(meta, direction + "_shapes")
        dtypes = getattr(meta, direction + "_dtypes")
        quants = getattr(meta, direction + "_quants")
        if len(names) != len(shapes) or set(names) != set(shapes):
            raise MetadataMismatchError(f"Exact named {direction} set required")
        result = {}
        for name, shape in shapes.items():
            if actual_shapes.get(name) != shape:
                raise MetadataMismatchError(f"Unexpected {direction} shape for {name}")
            result[name] = transform(
                dtypes.get(name),
                shape,
                quants.get(name),
                input_tensor=direction == "input",
            )
        transforms.append(MappingProxyType(result))
    return ModelBinding(selection, meta, *transforms)
