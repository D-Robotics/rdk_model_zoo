"""Exact S100 publication selection and fixed physical tensor contracts."""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from samples._shared.assets import Asset, list_assets
from samples._shared.runtime_meta import MetadataMismatchError, RuntimeMetadata

SAMPLE_DIR = Path(__file__).resolve().parents[2]
STAGES = ("encoder", "predictor", "decoder")
FILENAMES = {
    "encoder": "s100/paraformer_large_encoder_400x560_s100.hbm",
    "predictor": "s100/paraformer_large_predictor_400x512_s100.hbm",
    "decoder": "s100/paraformer_large_decoder_400x512_s100.hbm",
}
CONTEXT = "/encoder/after_norm/Add_1_output_0"

# Names/shapes follow the archived native lookup and ONNX extraction graphs.
# The lower-case acoustic alias is also handled by the source Python runtime.
INPUTS = {
    "encoder": {"features": (("speech",), (1, 400, 560), "float32")},
    "predictor": {"context": ((CONTEXT,), (1, 400, 512), "float32")},
    "decoder": {
        "context": ((CONTEXT,), (1, 400, 512), "float32"),
        "count": (("token_num",), (1,), "int32"),
        "bias": (("bias_embed",), (1, 1, 512), "float32"),
        "acoustic": (("onnx::Shape_8609", "shape_8609"), (1, 100, 512), "float32"),
    },
}
OUTPUTS = {
    "encoder": {"context": ((CONTEXT,), (1, 400, 512), "float32")},
    "predictor": {
        "alphas": (("/predictor/Add_output_0",), (1, 401), "float32"),
        "hidden": (("/predictor/Concat_5_output_0",), (1, 401, 512), "float32"),
    },
    "decoder": {"logits": (("logits",), (1, 100, 8404), "float32")},
}


@dataclass(frozen=True)
class Selection:
    target: str
    stage: str
    asset: Asset
    model_path: Path
    explicit_model_path: bool = False


@dataclass(frozen=True)
class Binding:
    selection: Selection
    metadata: RuntimeMetadata
    inputs: Mapping[str, str]
    outputs: Mapping[str, str]

    @property
    def model_name(self):
        return self.metadata.model_name


def resolve_selections(target="auto", *, model_paths=None, asset_ids=None):
    """Select all three models; alternate paths require all three asset IDs."""
    if target == "auto":
        from samples._shared.platforms import detect_target

        target = detect_target()
    if target != "s100":
        raise ValueError("Paraformer has a published three-model set only for s100")
    for name, values in (("model_paths", model_paths), ("asset_ids", asset_ids)):
        if values is not None and (
            not isinstance(values, Mapping) or set(values) != set(STAGES)
        ):
            raise ValueError(f"{name} requires exactly encoder, predictor and decoder")
    if model_paths is not None and asset_ids is None:
        raise ValueError("External model paths require explicit matching asset IDs")
    assets = {asset.filename: asset for asset in list_assets("s", "paraformer")}
    selected = []
    for stage in STAGES:
        asset = assets.get(FILENAMES[stage])
        if asset is None or asset.format != "hbm" or not asset.url:
            raise ValueError(f"Missing published {stage} HBM asset")
        if asset_ids is not None and asset_ids[stage] != asset.reference:
            raise ValueError(f"Expected {stage} asset-id {asset.reference}")
        path = (
            Path(model_paths[stage]).expanduser()
            if model_paths is not None
            else SAMPLE_DIR / "model" / asset.filename
        )
        selected.append(Selection(target, stage, asset, path, model_paths is not None))
    return tuple(selected)


def _roles(names, shapes, dtypes, contracts):
    if len(set(names)) != len(names):
        raise MetadataMismatchError("Physical tensor names must be unique")
    roles = {}
    for role, (aliases, shape, dtype) in contracts.items():
        matches = [name for name in names if name in aliases]
        if len(matches) != 1:
            raise MetadataMismatchError(f"Expected one exact physical name for {role}")
        name = matches[0]
        if shapes.get(name) != shape or dtypes.get(name) != dtype:
            raise MetadataMismatchError(f"{name} must be {dtype} {shape}")
        roles[role] = name
    if set(names) != set(roles.values()):
        raise MetadataMismatchError("Unexpected extra physical tensors")
    return roles


def validate_selection(selection):
    """Reject stage/asset/path mismatches before SDK construction."""
    if selection.stage not in STAGES:
        raise ValueError("Unknown Paraformer stage")
    expected = resolve_selections(selection.target)[STAGES.index(selection.stage)]
    if selection.asset != expected.asset or (
        not selection.explicit_model_path
        and selection.model_path != expected.model_path
    ):
        raise ValueError("Selection does not match the declared stage publication/path")


def bind_model(selection, metadata):
    """Validate publication identity, one model and every exposed I/O tensor."""
    validate_selection(selection)
    meta = (
        metadata
        if isinstance(metadata, RuntimeMetadata)
        else RuntimeMetadata.from_mapping(metadata)
    )
    if meta.model_names != (meta.model_name,):
        raise MetadataMismatchError(
            "Each Paraformer artifact must expose exactly one model"
        )
    inputs = _roles(
        meta.input_names, meta.input_shapes, meta.input_dtypes, INPUTS[selection.stage]
    )
    output_contract = dict(OUTPUTS[selection.stage])
    # The source ONNX decoder also exposes token_num. Some compiled interfaces
    # eliminate that pass-through output. Validate it when present, never by position.
    if selection.stage == "decoder" and "token_num" in meta.output_names:
        output_contract["count"] = (("token_num",), (1,), "int32")
    outputs = _roles(
        meta.output_names, meta.output_shapes, meta.output_dtypes, output_contract
    )
    return Binding(selection, meta, inputs, outputs)


def physical_inputs(binding):
    return {
        name: (binding.metadata.input_shapes[name], binding.metadata.input_dtypes[name])
        for name in binding.metadata.input_names
    }
