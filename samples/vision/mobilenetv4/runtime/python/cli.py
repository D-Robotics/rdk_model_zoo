"""Command-line surface for the MobileNetV4 sample.

Option declarations, the model-free listing/dry-run modes, and result
presentation live here so ``main.py`` can stay a thin, readable entry:
parse arguments, construct the model, call ``predict``, show the result.
Nothing in this module classifies images or loads a board SDK.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from samples.vision.mobilenetv4.runtime.python.labels import load_labels as _load_labels
from samples.vision.mobilenetv4.runtime.python.model_binding import (
    SUPPORTED_TARGETS,
    SUPPORTED_VARIANTS,
    AssetRecord,
    ModelSelection,
    list_available_assets,
    resolve_selection,
)

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
        choices=(0, 1),
        default=None,
        help="0 direct resize or 1 letterbox; default follows the bound source.",
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

    from samples.vision.mobilenetv4.runtime.python.model_binding import BindingError

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

    from samples.vision.mobilenetv4.runtime.python.model_binding import BindingError

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


def read_bgr_image(path: "str | Path"):
    """Read one BGR image; failures name the exact path."""

    import cv2

    resolved = Path(path).expanduser()
    image = cv2.imread(str(resolved), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"image not found or unreadable: {resolved}")
    return image


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
