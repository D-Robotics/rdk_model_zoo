"""Freeze MobileNet host inputs after verifying source and dataset bytes.

This preparation receipt is not a board acceptance or publication receipt.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from utils.py_utils.classification_host import load_dataset, sha256_file, write_json
from utils.tools.mobilenet.workflow import PINS, check_source, environment, provenance


def main() -> None:
    """Validate full manifests and write an immutable campaign directory."""
    from timm.data import ImageNetInfo

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--evaluation-manifest", type=Path, required=True)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--calibration-manifest", type=Path, required=True)
    parser.add_argument("--calibration-root", type=Path, required=True)
    parser.add_argument("--environment-lock", type=Path, required=True)
    parser.add_argument("--toolchains", type=Path, required=True)
    parser.add_argument("--boards-reported-ready", nargs="*", default=[],
                        choices=("x5", "s100", "s100p", "s600"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    pins = json.loads(PINS.read_text())
    for spec in pins["models"].values():
        check_source(args.source_root / spec["repo_id"].split("/")[-1] / spec["revision"], spec)
    evaluation, eval_images = load_dataset(args.evaluation_manifest, args.evaluation_root,
                                           expected_count=10000)
    calibration, cal_images = load_dataset(args.calibration_manifest, args.calibration_root,
                                           expected_count=200, labeled=False)
    if {r["sha256"] for _, r in eval_images} & {r["sha256"] for _, r in cal_images}:
        raise ValueError("Calibration and evaluation overlap")
    class_counts = {index: 0 for index in range(1000)}
    for _, row in eval_images:
        class_counts[row["label_id"]] += 1
    if set(class_counts.values()) != {10}:
        raise ValueError("ImageNetV2 matched-frequency requires ten images per class")
    args.output.mkdir(parents=True, exist_ok=False)
    copies = {"checkpoints.json": PINS, "evaluation-manifest.json": args.evaluation_manifest,
              "calibration-manifest.json": args.calibration_manifest,
              "environment.lock.txt": args.environment_lock, "toolchains.json": args.toolchains}
    for filename, source in copies.items():
        shutil.copyfile(source, args.output / filename)
    imagenet = ImageNetInfo()
    write_json(args.output / "labels.json", {"class_count": 1000, "classes": [
        {"id": index, "synset": imagenet.index_to_label_name(index),
         "description": imagenet.index_to_description(index)} for index in range(1000)]})
    files = {p.name: sha256_file(p) for p in sorted(args.output.iterdir())}
    write_json(args.output / "campaign.json", {
        "schema_version": 1, "campaign_id": args.output.name, "stage": "P0-host-inputs",
        "status": "frozen", "command": sys.argv, "source": provenance(),
        "freeze_script_sha256": sha256_file(Path(__file__)), "environment": environment(),
        "inputs": files, "source_models": pins["models"],
        "local_roots": {"source": str(args.source_root.resolve()),
                        "evaluation": str(args.evaluation_root.resolve()),
                        "calibration": str(args.calibration_root.resolve())},
        "evaluation": {"dataset": evaluation["dataset"], "count": len(eval_images),
                       "class_count": 1000, "images_per_class": 10,
                       "scope": "ImageNetV2 MatchedFrequency; not ILSVRC2012 val"},
        "calibration": {"dataset": calibration["dataset"], "count": len(cal_images),
                        "overlap_count": 0, "status": "frozen-candidate",
                        "suitability_gate": "P2 quantized full-set Top-1/Top-5 comparison"},
        "quality_gates": {"onnx_rtol": 1e-4, "onnx_atol": 1e-4, "smoke_top5_order_equal": True,
                          "max_top1_drop": 0.01, "max_top5_drop": 0.01,
                          "drop_unit": "absolute proportion (0.01 = 1 percentage point)",
                          "policy": "local campaign candidate gate; not an upstream model-zoo standard",
                          "baseline": "same dataset/geometry FP32 ONNX; includes NV12 differences in board delta"},
        "platforms": {name: {"availability": "user-reported-ready" if name in args.boards_reported_ready else "unknown",
                              "connection": "pending-user-P2-document",
                              "board_verification": "not-run"} for name in ("x5", "s100", "s100p", "s600")},
        "release_status": "not-released",
    })
    checksums = {p.name: sha256_file(p) for p in sorted(args.output.iterdir())}
    (args.output / "SHA256SUMS").write_text("".join(f"{digest}  {name}\n" for name, digest in checksums.items()))
    print(json.dumps({"campaign": str(args.output), "verified_evaluation_images": len(eval_images),
                      "verified_calibration_images": len(cal_images), "source_models": len(pins["models"])}))


if __name__ == "__main__":
    main()
