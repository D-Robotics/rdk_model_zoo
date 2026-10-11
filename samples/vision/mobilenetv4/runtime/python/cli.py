"""Select MobileNetV4 models, parse CLI options, and present results."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional

from utils.py_utils import cls_binding
from utils.py_utils.cls_binding import (  # noqa: F401 - re-exported surface
    OUTPUT_TRANSFORMS,
    AssetRecord,
    BindingError,
    ClassificationContract,
    KNOWN_OUTPUT_SEMANTICS,
    ManifestAssetError,
    ModelBinding,
    ModelSelection,
    RuntimeMetadata,
    SCORE_POLICIES,
    SUPPORTED_TARGETS as _SHARED_TARGETS,
    SampleBindingTable,
    UnsupportedAssetError,
    VariantFacts,
    contract_input_is_packed,
    normalise_score_vector,
    score_vector_shape,
)
from utils.py_utils.cls_binding import MetadataMismatchError  # noqa: F401
from utils.py_utils.platform_profile import (
    PlatformProfile,
    UnsupportedProfileError,
    classification_profiles,
    resolve_profile,
)


#: Targets the shared classification machinery can address.
SUPPORTED_TARGETS = _SHARED_TARGETS
#: MobileNetV4 variants published across the manifests.
SUPPORTED_VARIANTS = ('small', 'medium', 'large')
_SAMPLE_DIR = Path(__file__).resolve().parents[2]

#: Platform deployment profiles for this sample (H5).  X5 publishes flat
#: ``.bin`` artifacts; S100, S100P and S600 publish ``.hbm`` artifacts under
#: ``s100/``, ``s100p/`` and ``s600/``.
PLATFORMS = classification_profiles(
    url_prefix_s="rdk_s100/MobileNet",
)

#: Published variants: name -> (square input size, shorter-edge resize).  The
#: resize is ``int(size / crop_pct)`` of the pinned timm checkpoint.
_VARIANTS = {
    'small': (224, 256),  # mobilenetv4_conv_small
    'medium': (224, 235),  # mobilenetv4_conv_medium
    'large': (256, 269),  # mobilenetv4_conv_large
}


def _crop_facts(size: int, resize_shorter: int) -> VariantFacts:
    """Return the contract for a square model fed by shorter-edge center crop.

    ``resize_interpolation`` is unused by this policy, which always resizes
    with antialiased PIL bicubic.
    """

    return VariantFacts(
        input_height=size,
        input_width=size,
        output_score_policy="softmax",
        output_semantics="source_declared_logits",
        resize_type=2,  # shorter-edge resize + center crop (timm evaluation)
        resize_interpolation="cubic",
        resize_shorter=resize_shorter,
    )


_FACTS = {
    (variant, target): _crop_facts(size, shorter)
    for variant, (size, shorter) in _VARIANTS.items()
    for target in SUPPORTED_TARGETS
}

BINDING_TABLE = SampleBindingTable(
    sample_dir=_SAMPLE_DIR,
    manifest_rows=(
        ('x5', 'mobilenetv4'),
        ('s', 'mobilenetv4'),
    ),
    filename_variants={
        'mobilenetv4_conv_small_bayese_224x224_nv12.bin': 'small',
        's100/mobilenetv4_conv_small_nashe_224x224_nv12.hbm': 'small',
        's100p/mobilenetv4_conv_small_nashm_224x224_nv12.hbm': 'small',
        's600/mobilenetv4_conv_small_nashp_224x224_nv12.hbm': 'small',
        'mobilenetv4_conv_medium_bayese_224x224_nv12.bin': 'medium',
        's100/mobilenetv4_conv_medium_nashe_224x224_nv12.hbm': 'medium',
        's100p/mobilenetv4_conv_medium_nashm_224x224_nv12.hbm': 'medium',
        's600/mobilenetv4_conv_medium_nashp_224x224_nv12.hbm': 'medium',
        'mobilenetv4_conv_large_bayese_256x256_nv12.bin': 'large',
        's100/mobilenetv4_conv_large_nashe_256x256_nv12.hbm': 'large',
        's100p/mobilenetv4_conv_large_nashm_256x256_nv12.hbm': 'large',
        's600/mobilenetv4_conv_large_nashp_256x256_nv12.hbm': 'large',
    },
    default_variant='small',
    facts=_FACTS,
    s_filename_targets=('s100', 's100p', 's600'),
)


def list_available_assets(target: Optional[str] = None) -> tuple[AssetRecord, ...]:
    """Return the finite sample assets read from the existing manifests.

    ``target=None`` or ``target="auto"`` is intentionally host-independent so
    the listing command can run on a workstation.
    """

    return cls_binding.list_assets(BINDING_TABLE, target)


def resolve_selection(
    target: str = "auto",
    *,
    asset_id: Optional[str] = None,
    variant: Optional[str] = None,
    model_path: Optional[str | Path] = None,
    soc_name: Optional[str] = None,
    board_type: Optional[str] = None,
) -> ModelSelection:
    """Resolve one published MobileNetV4 asset and its source-proven contract.

    ``model_path`` is accepted only with an exact qualified manifest reference.
    """

    return cls_binding.resolve_selection(
        BINDING_TABLE,
        target,
        asset_id=asset_id,
        variant=variant,
        model_path=model_path,
        soc_name=soc_name,
        board_type=board_type,
    )


def bind_model(
    selection: ModelSelection, metadata: RuntimeMetadata | Mapping[str, Any]
) -> ModelBinding:
    """Validate actual runtime metadata against the MobileNetV4 contract table."""

    return cls_binding.bind_model(BINDING_TABLE, selection, metadata)




import argparse
from pathlib import Path
from utils.py_utils.image import read_bgr_image
import sys

from utils.py_utils.labels import load_labels as _load_labels

_ROOT = Path(__file__).resolve().parents[5]
_SAMPLE_DIR = _ROOT / "samples" / "vision" / "mobilenetv4"
_DEFAULT_IMAGE = _SAMPLE_DIR / "test_data" / "great_grey_owl.JPEG"
# Root datasets/ is the A2 unified location; the per-platform snapshot paths
# under platforms/ disappear with the migration closeout.
_DEFAULT_LABELS = _ROOT / "datasets" / "imagenet" / "imagenet_classes.names"


def build_parser() -> argparse.ArgumentParser:
    """Build the SDK-free command line parser for the sample entrypoint."""

    parser = argparse.ArgumentParser(
        description="MobileNetV4 ImageNet classification on an observed RDK target."
    )
    parser.add_argument(
        "--target",
        choices=("auto",) + SUPPORTED_TARGETS,
        default="auto",
        help="Execution target (auto, x5, s100, s100p, or s600).",
    )
    parser.add_argument(
        "--asset-id",
        help="Exact manifest reference group:sample:filename; see --list-models.",
    )
    parser.add_argument(
        "--variant",
        choices=SUPPORTED_VARIANTS,
        default=None,
        help="Model variant (default: small; see --list-models for the "
        "published variant/target combinations).",
    )
    parser.add_argument(
        "--model-path",
        help="Path to an existing compiled artifact; no download is performed.",
    )
    parser.add_argument(
        "--test-img",
        default=str(_DEFAULT_IMAGE),
        help="BGR input image path (default: bundled great_grey_owl.JPEG test image).",
    )
    parser.add_argument(
        "--label-file",
        default=str(_DEFAULT_LABELS),
        help="One-label-per-line ImageNet labels file.",
    )
    parser.add_argument(
        "--top-k",
        "--topk",
        dest="top_k",
        type=int,
        default=5,
        help="Number of results to print (default: 5).",
    )
    parser.add_argument(
        "--resize-type",
        type=int,
        choices=(0, 1, 2),
        default=None,
        help="0 direct resize, 1 letterbox, or 2 shorter-edge resize + center crop "
        "(needs Pillow); default follows the bound model (2).",
    )
    parser.add_argument(
        "--priority",
        type=int,
        default=0,
        help="Runtime scheduling priority (0-255; default: 0).",
    )
    parser.add_argument(
        "--bpu-cores",
        nargs="+",
        type=int,
        default=[0],
        help="Runtime BPU core indexes (default: 0).",
    )
    parser.add_argument(
        "--img-save-path",
        help="Optional path for a simple annotated result image.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--list-models",
        action="store_true",
        help="List manifest-backed sample asset references without board access.",
    )
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve/check a selection without loading a model or SDK.",
    )
    return parser


def run_list_models(target: str) -> int:
    """Print the manifest-backed references for ``target`` (model-free)."""

    try:
        records = list_available_assets(target)
    except BindingError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print("Manifest-backed MobileNetV4 asset references:")
    if not records:
        print(f"  no published asset for target: {target}")
        return 0
    for record in records:
        _print_record(record)
    print("Asset IDs above are qualified manifest references, not standalone catalog IDs.")
    return 0


def run_dry_run(args: argparse.Namespace) -> int:
    """Print a concrete contract without detecting hardware or loading SDK."""

    try:
        if args.target == "auto":
            candidates = list_available_assets("auto")
            if args.asset_id is not None:
                candidates = tuple(
                    record for record in candidates if record.asset_id == args.asset_id
                )
            variant = getattr(args, "variant", None)
            if variant is not None:
                candidates = tuple(
                    record for record in candidates if record.variant == variant
                )
            if len(candidates) != 1:
                print("Dry-run needs an explicit target or one qualified asset reference.")
                print("Candidates:")
                for record in candidates:
                    _print_record(record)
                print("No model is downloaded and no SDK is loaded.")
                return 0 if candidates else 2
            record = candidates[0]
            selection = resolve_selection(
                record.target,
                asset_id=record.asset_id,
                model_path=args.model_path,
            )
        else:
            selection = resolve_selection(
                args.target,
                asset_id=args.asset_id,
                variant=getattr(args, "variant", None),
                model_path=args.model_path,
            )
    except BindingError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print("Dry-run selection:")
    print(f"  target: {selection.target}")
    print(f"  asset_id: {selection.asset_id}")
    print(f"  sample_id: {selection.sample_id}")
    print(f"  variant: {selection.variant}")
    print(f"  model_path: {selection.model_path}")
    print(f"  model_format: {selection.contract.model_format}")
    print(f"  input_protocol: {selection.contract.input_protocol}")
    print(
        "  input_geometry: "
        f"{selection.contract.input_width}x{selection.contract.input_height}"
    )
    print(
        f"  preprocess: resize_type={selection.contract.resize_type}, "
        f"resize_shorter={selection.contract.resize_shorter}"
    )
    print(f"  output_transform: {selection.contract.output_transform}")
    print(f"  output_rank_rule: squeeze -> ({selection.contract.class_count},)")
    print(f"  output_semantics: {selection.contract.output_semantics}")
    print(f"  output_score_policy: {selection.contract.output_score_policy}")
    print(f"  source_manifest: {selection.contract.source_manifest}")
    print(f"  model_path_exists: {selection.model_path.is_file()}")
    print("No model is downloaded and no SDK is loaded.")
    return 0


def default_labels(label_file: str):
    """Load the labels named by ``--label-file``.

    The parser default points at the bundled label source that matches the
    published class count; an explicit file always wins, and its coverage
    of the bound class count is checked when the classifier is constructed,
    so a mismatched file fails with a concrete error instead of
    mislabeling results.
    """

    return _load_labels(Path(label_file).expanduser())




def print_classification_result(result, selection: ModelSelection, *,
                                top_k: int) -> None:
    """Print the Top-K lines for one finished prediction."""

    print(f"Top-{top_k} results ({selection.target}, {selection.asset_id}):")
    for rank, (class_id, score, label) in enumerate(
        zip(result.class_ids.tolist(), result.scores.tolist(), result.labels), start=1
    ):
        print(f"  Rank {rank}: class={class_id}, label={label}, score={score:.6f}")


def save_result_image(path: Path, image, result) -> None:
    """Write a simple annotated copy of the input image (presentation only)."""

    import cv2

    canvas = image.copy()
    y = 28
    for rank, (class_id, score, label) in enumerate(
        zip(result.class_ids, result.scores, result.labels), start=1
    ):
        cv2.putText(
            canvas,
            f"{rank}: {int(class_id)} {label} {float(score):.4f}",
            (8, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )
        y += 22
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), canvas):
        raise OSError(f"failed to save result image: {path}")


def _print_record(record: AssetRecord) -> None:
    print(f"  asset_id: {record.asset_id}")
    print(f"    target: {record.target}")
    print(f"    sample_id: {record.sample_id}")
    print(f"    variant: {record.variant}")
    print(f"    filename: {record.filename}")
    print(f"    format: {record.model_format}")
    print(f"    source_manifest: {record.source_manifest}")


__all__ = [
    "build_parser",
    "default_labels",
    "print_classification_result",
    "read_bgr_image",
    "run_dry_run",
    "run_list_models",
    "save_result_image",
]
