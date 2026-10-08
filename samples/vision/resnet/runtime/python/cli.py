"""Provide ResNet command options, model selection, and result presentation.

The entry point uses these helpers to list artifacts, preview a selection,
and print or save classification results. Model inference lives in classify.
"""


from pathlib import Path
import argparse

from utils.py_utils import cls_binding
from utils.py_utils.cls_binding import SUPPORTED_TARGETS, SampleBindingTable, VariantFacts
from samples.vision.resnet.model.download import ASSET_REFERENCES, VARIANTS as SUPPORTED_VARIANTS

_ROOT = Path(__file__).resolve().parents[5]
_DEFAULT_IMAGE = _ROOT / "samples/vision/resnet/test_data/white_wolf.JPEG"

# Published choices reuse the downloader's exact artifact references.
BINDING_TABLE = SampleBindingTable(
    sample_dir=_ROOT / "samples/vision/resnet",
    manifest_rows=tuple(dict.fromkeys(tuple(ref.split(":", 2)[:2]) for ref in ASSET_REFERENCES.values())),
    filename_variants={ref.split(":", 2)[2]: variant for (_, variant), ref in ASSET_REFERENCES.items()},
    default_variant="resnet18",
    facts={
        (variant, target): VariantFacts(
            input_height=224, input_width=224,
            resize_interpolation="linear" if target == "x5" else "nearest",
            output_semantics="unverified_score_vector" if variant == "resnet18" else "source_declared_logits",
            output_score_policy="legacy_softmax" if variant == "resnet18" else "softmax")
        for target, variant in ASSET_REFERENCES
    },
)


def resolve_selection(target="auto", **options):
    """Resolve the published artifact and contract for a ResNet command.

    Args:
        target: x5, s100, s100p, s600, or auto to detect the executing board.
        **options: Optional asset_id, variant, and model_path selection arguments
            forwarded to the shared resolver.

    Returns:
        ModelSelection: Concrete target, artifact path, and tensor/output contract.

    Raises:
        BindingError: No unique supported artifact matches the requested selection.
        ValueError: The target is unknown or automatic board detection fails.
    """
    return cls_binding.resolve_selection(BINDING_TABLE, target, **options)


def list_available_assets(target=None):
    """Read published ResNet artifacts without loading the board SDK.

    Args:
        target: Optional concrete target filter; None or auto includes all targets.

    Returns:
        tuple[AssetRecord, ...]: Published artifacts in manifest order.

    Raises:
        ValueError: The target or a manifest-backed record is invalid.
    """
    return cls_binding.list_assets(BINDING_TABLE, target)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser with ResNet defaults.

    Returns:
        argparse.ArgumentParser: Parser for model selection, input, scheduling,
        output, and model-free listing/dry-run options.
    """
    parser = argparse.ArgumentParser(description="ResNet image classification on RDK.")
    parser.add_argument("--target", choices=("auto",) + SUPPORTED_TARGETS, default="auto",
                        help="Execution board; auto detects the local hardware.")
    parser.add_argument("--asset-id", help="Qualified model reference from --list-models.")
    parser.add_argument("--variant", choices=SUPPORTED_VARIANTS,
                        help="Default: resnet18. resnet50/resnet152: S100/S600.")
    parser.add_argument("--model-path", help="Compiled model path; requires --asset-id.")
    parser.add_argument("--test-img", default=str(_DEFAULT_IMAGE), help="Input image path.")
    parser.add_argument("--label-file", help="Class names file; default: bundled ImageNet labels.")
    parser.add_argument("--top-k", "--topk", dest="top_k", type=int, default=5)
    parser.add_argument("--resize-type", type=int, choices=(0, 1),
                        help="0 stretch, 1 letterbox; default follows the model contract.")
    parser.add_argument("--priority", type=int, default=0, help="Scheduling priority, 0–255.")
    parser.add_argument("--bpu-cores", nargs="+", type=int, default=[0], help="BPU core indexes.")
    parser.add_argument("--img-save-path", help="Save an annotated result image.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--list-models", action="store_true", help="List published models.")
    mode.add_argument("--dry-run", action="store_true", help="Show selection without loading the SDK.")
    return parser


def list_models(target: str) -> int:
    """Print published ResNet artifacts for the requested target.

    Args:
        target: Concrete target or auto to list all published targets.

    Returns:
        int: 0 after printing the list, including when it is empty.

    Raises:
        ValueError: The target or an artifact record is invalid.
    """
    records = list_available_assets(target)
    print("Manifest-backed ResNet asset references:")
    for record in records:
        print_record(record)
    if not records:
        print(f"  no published asset for target: {target}")
    return 0


def dry_run(args) -> int:
    """Print a model selection without detecting hardware or loading the SDK.

    Args:
        args: Parsed namespace containing target, asset_id, variant, and model_path.

    Returns:
        int: 0 for a resolved selection or a nonempty candidate list; 2 if no
        candidates match an automatic selection.

    Raises:
        BindingError: Explicit artifact selection is invalid.

    Notes:
        With target=auto, only a unique candidate resolves to a concrete target.
        This command does not download or execute the model.
    """
    target, asset_id = args.target, args.asset_id
    if target == "auto":
        candidates = [r for r in list_available_assets("auto")
                      if (asset_id is None or r.asset_id == asset_id)
                      and (args.variant is None or r.variant == args.variant)]
        if len(candidates) != 1:
            print("Dry-run needs an explicit target or one qualified asset reference.")
            print("Candidates:")
            for record in candidates:
                print_record(record)
            print("No model is downloaded and no SDK is loaded.")
            return 0 if candidates else 2
        target, asset_id = candidates[0].target, candidates[0].asset_id
    selection = resolve_selection(target, asset_id=asset_id, variant=args.variant, model_path=args.model_path)
    contract = selection.contract
    details = {name: getattr(selection, name) for name in
               ("target", "asset_id", "sample_id", "variant", "model_path")}
    details.update({name: getattr(contract, name) for name in
                    ("model_format", "input_protocol", "output_transform",
                     "output_semantics", "output_score_policy", "source_manifest")})
    details.update(input_geometry=f"{contract.input_width}x{contract.input_height}",
                   output_rank_rule=f"squeeze -> ({contract.class_count},)",
                   model_path_exists=selection.model_path.is_file())
    print("Dry-run selection:")
    for name, value in details.items():
        print(f"  {name}: {value}")
    print("No model is downloaded and no SDK is loaded.")
    return 0


def print_record(record) -> None:
    """Print the identity and format of one published artifact.

    Args:
        record: AssetRecord with target, variant, filename, and source manifest.

    Returns:
        None.
    """
    print(f"  asset_id: {record.asset_id}")
    for name in ("target", "sample_id", "variant", "filename", "source_manifest"):
        print(f"    {name}: {getattr(record, name)}")
    print(f"    format: {record.model_format}")


def save_result_image(path: Path, image, result) -> None:
    """Save a copy of the input image with ranked classification labels.

    Args:
        path: Destination image path; missing parent directories are created.
        image: uint8 BGR image shaped (H, W, 3), values [0, 255]; not modified.
        result: ClassificationResult with matching class_ids, scores, and labels.

    Returns:
        None.

    Raises:
        OSError: The output directory or image cannot be written.
        cv2.error: OpenCV cannot encode the requested image format or draw the image.
    """
    import cv2

    canvas = image.copy()
    for rank, (class_id, score, label) in enumerate(
        zip(result.class_ids, result.scores, result.labels), start=1
    ):
        cv2.putText(canvas, f"{rank}: {int(class_id)} {label} {float(score):.4f}",
                    (8, 28 + (rank - 1) * 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 0, 255), 1, cv2.LINE_AA)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), canvas):
        raise OSError(f"failed to save result image: {path}")



def print_result(result, selection, top_k):
    """Print ranked class IDs, names, and scores for a selected artifact.

    Args:
        result: ClassificationResult containing equally sized ranked fields.
        selection: ModelSelection supplying the target and artifact identifier.
        top_k: Number of results used in the output heading.

    Returns:
        None.
    """
    print(f"Top-{top_k} results ({selection.target}, {selection.asset_id}):")
    for rank, (class_id, score, label) in enumerate(
        zip(result.class_ids, result.scores, result.labels), start=1
    ):
        print(f"  Rank {rank}: class={class_id}, label={label}, score={score:.6f}")
