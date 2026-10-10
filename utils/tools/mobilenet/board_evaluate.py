"""Evaluate a compiled MobileNetV4 model on the frozen full ImageNetV2 manifest.

The host's PIL geometry is applied before the sample's NV12 conversion. Model
loading happens once. This accuracy run does not claim performance metrics.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import PIL

from samples.vision.mobilenetv4.runtime.python.classify import MobileNetV4Classifier
from utils.py_utils.classification_host import (
    load_dataset, prepare_rgb, sha256_file, write_json,
)


def evaluate(args: argparse.Namespace) -> None:
    """Validate every dataset hash and evaluate the requested board artifact.

    Args:
        args: Explicit model, target, campaign, manifest, data and output paths.

    Raises:
        ValueError: Input hashes, model metadata or dataset coverage disagree.
    """
    campaign = json.loads(args.campaign.read_text())
    contract = campaign["source_models"][args.variant]["contract"]
    if sha256_file(args.model) != args.model_sha256:
        raise ValueError("Model SHA256 mismatch")
    args.output.mkdir(parents=True, exist_ok=False)
    _, images = load_dataset(args.manifest, args.data_root,
                             expected_count=campaign["evaluation"]["count"])
    classifier = MobileNetV4Classifier(args.model, target=args.target,
                                      resize_type=0, score_policy="none")
    metadata = classifier.runner.metadata
    # SDK QuantParams are pybind objects, not serializable dataclasses.
    write_json(args.output / "metadata.json", {
        name: getattr(metadata, name) for name in (
            "model_name", "model_names", "input_names", "input_shapes", "input_dtypes",
            "input_strides", "output_names", "output_shapes", "output_dtypes", "output_strides")
    })
    started = datetime.now(timezone.utc).isoformat()
    clock = time.monotonic()
    counts = [0, 0]
    smoke = []
    with (args.output / "predictions.jsonl").open("x") as stream:
        for index, (path, record) in enumerate(images):
            rgb = prepare_rgb(path, contract)
            prepared = classifier.preprocess(np.ascontiguousarray(rgb[:, :, ::-1]))
            outputs = classifier.infer(prepared)
            result = classifier.postprocess(outputs)
            ids = result.class_ids.tolist()
            label = record["label_id"]
            counts[0] += ids[0] == label
            counts[1] += label in ids
            prediction = {"path": record.get("path", record.get("name")),
                          "sha256": record["sha256"], "label_id": label,
                          "top5_ids": ids, "scores": result.scores.tolist(),
                          "rgb_sha256": hashlib.sha256(rgb.tobytes()).hexdigest()}
            stream.write(json.dumps(prediction) + "\n")
            if index < 3:
                np.save(args.output / f"smoke-{index}-rgb.npy", rgb)
                for name, data in outputs.items():
                    np.save(args.output / f"smoke-{index}-logits.npy", data)
                smoke.append(prediction)
            if (index + 1) % 1000 == 0:
                print(json.dumps({"count": index + 1, "correct": counts}), flush=True)
    report = {"schema_version": 1, "stage": "board-accuracy", "target": args.target,
              "variant": args.variant,
              "model_sha256": args.model_sha256, "campaign_sha256": sha256_file(args.campaign),
              "manifest_sha256": sha256_file(args.manifest), "count": len(images),
              "top1_correct": counts[0], "top5_correct": counts[1],
              "top1": counts[0] / len(images), "top5": counts[1] / len(images),
              "dataset": campaign["evaluation"], "contract": contract,
              "started_at": started, "elapsed_seconds": time.monotonic() - clock,
              "environment": {"python": sys.version, "os": platform.platform(),
                              "numpy": np.__version__, "Pillow": PIL.__version__,
                              "opencv": cv2.__version__},
              "command": sys.argv, "smoke": smoke,
              "source_sha256": {str(p.relative_to(ROOT)): sha256_file(p) for p in
                                (Path(__file__), ROOT / "utils/py_utils/classification_host.py")},
              "predictions_sha256": sha256_file(args.output / "predictions.jsonl")}
    write_json(args.output / "evaluation.json", report)
    print(json.dumps({k: report[k] for k in ("target", "count", "top1", "top5")}))


def main() -> None:
    """Parse a complete reproducible board accuracy invocation."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "campaign", "manifest", "data-root", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--model-sha256", required=True)
    parser.add_argument("--target", choices=("x5", "s100", "s100p", "s600"), required=True)
    parser.add_argument("--variant", choices=("v4-small", "v4-medium-224"), default="v4-small",
                        help="Campaign source model whose preprocessing contract applies")
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
