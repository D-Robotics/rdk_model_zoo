# Copyright (c) 2026 D-Robotics Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""People and Agent entrypoint for the ViT classification sample.

Only this thin entrypoint adjusts ``sys.path`` for a direct full-checkout
invocation.  Reusable algorithms use absolute package imports and do not alter
the import path or load a board SDK.  Listing and dry-run are deliberately
model-free; model preparation remains an explicit user action.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Iterable, Optional, Sequence


_ROOT = Path(__file__).resolve().parents[5]
if str(_ROOT) not in sys.path:
    # Direct ``python samples/.../main.py`` is a supported full-checkout entry.
    # This compatibility adjustment is kept at the entry boundary only.
    sys.path.insert(0, str(_ROOT))

from samples.vision.vit.runtime.python.model_binding import (  # noqa: E402
    BindingError,
    SUPPORTED_TARGETS,
    SUPPORTED_VARIANTS,
    AssetRecord,
    ModelSelection,
    list_available_assets,
    resolve_selection,
)
from samples.vision.vit.runtime.python.labels import load_labels as _load_labels  # noqa: E402


_SAMPLE_DIR = _ROOT / "samples" / "vision" / "vit"
_DEFAULT_IMAGE = _SAMPLE_DIR / "test_data" / "airplane_0000.png"
# CIFAR-10 labels are sample-specific; do not use ImageNet labels.
_DEFAULT_LABELS = _SAMPLE_DIR / "test_data" / "cifar10_classes.names"


def build_parser() -> argparse.ArgumentParser:
    """Build the SDK-free command line parser for the sample entrypoint."""

    parser = argparse.ArgumentParser(
        description="ViT CIFAR-10 classification on an observed RDK target."
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
        "--variant", "--model-variant",
        choices=SUPPORTED_VARIANTS,
        default=None,
        help="Model variant (default: int8; see --list-models for the "
        "published variant/target combinations).",
    )
    parser.add_argument(
        "--model-path",
        help="Path to an existing compiled artifact; no download is performed.",
    )
    parser.add_argument(
        "--test-img",
        default=str(_DEFAULT_IMAGE),
        help="BGR input image path (default: bundled airplane_0000.png test image).",
    )
    parser.add_argument(
        "--label-file",
        default=str(_DEFAULT_LABELS),
        help="CIFAR-10 label dictionary literal or one label per line.",
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


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the selected mode and return zero on success, two on user/runtime error."""

    parser = build_parser()
    args = parser.parse_args(argv)
    if args.list_models:
        return _list_models(args.target)
    if args.dry_run:
        return _dry_run(args)
    try:
        selection = resolve_selection(
            args.target,
            asset_id=args.asset_id,
            variant=getattr(args, "variant", None),
            model_path=args.model_path,
        )
        if not selection.model_path.is_file():
            raise FileNotFoundError(
                f"model file not found: {selection.model_path}"
            )

        # A path existing on disk is not execution authorization.  Require the
        # exact detected board before touching hbm_runtime.
        from samples._shared.platforms import require_execution_target

        require_execution_target(selection.target)
        return _run(selection, args)
    except (BindingError, FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


def _list_models(target: str) -> int:
    try:
        records = list_available_assets(target)
    except BindingError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print("Manifest-backed ViT asset references:")
    if not records:
        print(f"  no published asset for target: {target}")
        return 0
    for record in records:
        _print_record(record)
    print("Asset IDs above are qualified manifest references, not standalone catalog IDs.")
    return 0


def _dry_run(args: argparse.Namespace) -> int:
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
    print(f"  output_transform: {selection.contract.output_transform}")
    print(f"  output_rank_rule: squeeze -> ({selection.contract.class_count},)")
    print(f"  output_semantics: {selection.contract.output_semantics}")
    print(f"  output_score_policy: {selection.contract.output_score_policy}")
    print(f"  source_manifest: {selection.contract.source_manifest}")
    print(f"  model_path_exists: {selection.model_path.is_file()}")
    print("No model is downloaded and no SDK is loaded.")
    return 0


def _run(selection: ModelSelection, args: argparse.Namespace) -> int:
    # These imports are intentionally inside real execution.  They pull in
    # OpenCV and hbm_runtime only after selection, file, and board checks pass.
    import cv2
    import numpy as np

    from samples.vision.vit.runtime.python.classification import ClassificationTask
    from samples.vision.vit.runtime.python.model_runner import RuntimeModelRunner

    image_path = Path(args.test_img).expanduser()
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"test image not found or unreadable: {image_path}")
    labels = _load_labels(Path(args.label_file).expanduser())

    runner = RuntimeModelRunner(selection)
    binding = runner.load()
    runner.set_scheduling_params(priority=args.priority, bpu_cores=args.bpu_cores)
    task = ClassificationTask(
        runner,
        binding,
        top_k=args.top_k,
        labels=labels,
        resize_type=args.resize_type,
    )
    result = task.predict(np.asarray(image))
    print(f"Top-{args.top_k} results ({selection.target}, {selection.asset_id}):")
    for rank, (class_id, score, label) in enumerate(
        zip(result.class_ids.tolist(), result.scores.tolist(), result.labels), start=1
    ):
        print(f"  Rank {rank}: class={class_id}, label={label}, score={score:.6f}")

    if args.img_save_path:
        _save_visualization(
            Path(args.img_save_path).expanduser(), image, result.class_ids, result.scores, result.labels
        )
        print(f"Saved result image: {args.img_save_path}")
    return 0


def _save_visualization(
    path: Path,
    image,
    class_ids: Iterable[int],
    scores: Iterable[float],
    labels: Iterable[str],
) -> None:
    import cv2

    canvas = image.copy()
    y = 28
    for rank, (class_id, score, label) in enumerate(
        zip(class_ids, scores, labels), start=1
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


if __name__ == "__main__":
    raise SystemExit(main())
