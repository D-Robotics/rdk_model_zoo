# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Export a local YOLOE PF checkpoint and compare its ten outputs on the CPU."""

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from utils.py_utils.assets import sha256_file
from samples.vision.yoloe.model.vocabulary import LABELS_SHA256

VARIANTS = ("11s", "11m", "11l", "26n", "26s", "26m", "26l", "26x")


def export_checkpoint(*, weights, variant, output_dir, test_image=None, threads=2):
    """Write a fresh export directory; no downloads, compiler, board or target claim."""
    if variant not in VARIANTS:
        raise ValueError(f"Unknown PF variant: {variant}")
    if isinstance(threads, bool) or not isinstance(threads, int) or threads < 1:
        raise ValueError("threads must be a positive integer.")
    weights = Path(weights).expanduser().resolve()
    root = Path(output_dir).expanduser().resolve()
    if not weights.is_file():
        raise FileNotFoundError(f"Local checkpoint required: {weights}")
    if root.exists():
        raise FileExistsError(f"Refusing existing export directory: {root}")
    import torch
    import onnxruntime as ort
    import numpy as np
    import ultralytics
    from ultralytics import YOLOE
    from ultralytics.nn.modules.head import YOLOESegment, YOLOESegment26
    from samples.vision.yoloe.conversion.export_heads import (
        RawPFModel,
        reference_outputs,
        OUTPUT_NAMES,
    )
    from samples.vision.yoloe.conversion.contract import inspect_graph
    from samples.vision.yoloe.conversion.calibration import calibration_tensor
    from utils.py_utils.yoloe26_geometry import prepare_rgb

    if ultralytics.__version__ != "8.4.127":
        raise ValueError("Use the pinned export environment: ultralytics==8.4.127.")
    checkpoint_sha = sha256_file(weights)
    root.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "exporting",
        "variant": variant,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(weights),
        "checkpoint_sha256": checkpoint_sha,
        "versions": {
            "torch": torch.__version__,
            "ultralytics": ultralytics.__version__,
            "onnxruntime": ort.__version__,
        },
        "validation": {
            "upstream_static_pf": "not-run",
            "onnx_float": "not-run",
            "onnx_graph": "not-run",
            "compiled_model": "not-run",
            "board": "not-run",
            "dataset_accuracy": "not-run",
        },
    }
    original_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(threads)
        network = YOLOE(str(weights)).model.cpu().float().eval()
        if sha256_file(weights) != checkpoint_sha:
            raise ValueError("Checkpoint changed while loading.")
        expected_type = YOLOESegment26 if variant.startswith("26") else YOLOESegment
        if type(network.model[-1]) is not expected_type:
            raise ValueError(f"Checkpoint head does not match variant {variant}.")
        if network.yaml.get("scale") != variant[-1]:
            raise ValueError(
                f'Checkpoint scale {network.yaml.get("scale")!r} does not match variant {variant}.'
            )
        names = network.names
        if isinstance(names, dict):
            if set(names) != set(range(4585)):
                raise ValueError("PF class IDs must be contiguous 0..4584.")
            names = [names[i] for i in range(4585)]
        if len(names) != 4585 or any(not isinstance(name, str) for name in names):
            raise ValueError("Expected 4585 checkpoint-ordered PF labels.")
        labels = ("\n".join(names) + "\n").encode("utf-8")
        if sha256(labels).hexdigest() != LABELS_SHA256:
            raise ValueError(
                "Checkpoint vocabulary differs from the fixed PF vocabulary."
            )
        names_path = root / f"yoloe_{variant}_seg_pf.names"
        names_path.write_bytes(labels)
        wrapper = RawPFModel(network, variant).eval()
        if test_image is None:
            sample = torch.rand(
                (1, 3, 640, 640), generator=torch.Generator().manual_seed(0)
            )
            report["validation_input"] = {"kind": "seeded_random", "seed": 0}
        else:
            import cv2

            image_path = Path(test_image).expanduser().resolve()
            image_bytes = image_path.read_bytes()
            image = (
                cv2.imdecode(
                    np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR
                )
                if image_bytes
                else None
            )
            if image is None:
                raise ValueError(f"Unreadable test image: {image_path}")
            tensor = (
                prepare_rgb(image)
                if variant.startswith("26")
                else calibration_tensor(image, "x5", variant) / 255
            )
            sample = torch.from_numpy(tensor)
            report["validation_input"] = {
                "kind": "image",
                "path": str(image_path),
                "sha256": sha256(image_bytes).hexdigest(),
            }
        report["validation_input"]["tensor_sha256"] = sha256(
            sample.numpy().tobytes()
        ).hexdigest()
        onnx_path = root / f"yoloe_{variant}_seg_pf.onnx"
        with torch.inference_mode():
            report["validation"]["upstream_static_pf"] = "running"
            expected, comparison = reference_outputs(wrapper, sample)
            report["validation"]["upstream_comparison"] = comparison
            report["validation"]["upstream_static_pf"] = "passed"
            report["validation"]["onnx_graph"] = "running"
            torch.onnx.export(
                wrapper,
                (sample,),
                str(onnx_path),
                input_names=["images"],
                output_names=list(OUTPUT_NAMES),
                opset_version=17 if variant.startswith("26") else 11,
                dynamo=False,
                do_constant_folding=True,
                external_data=False,
            )
        facts = inspect_graph(onnx_path, names_path, variant)
        report["graph"] = facts
        report["validation"]["onnx_graph"] = "passed"
        report["validation"]["onnx_float"] = "running"
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        # Validate the exported graph itself. CPU graph fusions can add numerical
        # drift (observed on E11m); optimized-engine acceptance is a separate task.
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        report["validation"]["onnx_graph_optimization"] = "ORT_DISABLE_ALL"
        session = ort.InferenceSession(
            str(onnx_path), options, providers=["CPUExecutionProvider"]
        )
        actual = session.run(list(OUTPUT_NAMES), {"images": sample.numpy()})
        errors = {}
        report["validation"]["onnx_max_abs_error"] = errors
        for name, reference, result in zip(OUTPUT_NAMES, expected, actual, strict=True):
            reference = reference.numpy()
            if result.shape != reference.shape or result.dtype != np.float32:
                raise ValueError(f"ONNX output shape/dtype mismatch: {name}")
            if not np.isfinite(result).all() or not np.isfinite(reference).all():
                raise ValueError(f"Non-finite float output: {name}")
            errors[name] = float(np.max(np.abs(result - reference)))
            np.testing.assert_allclose(
                result, reference, rtol=2e-3, atol=2e-3, err_msg=name
            )
        report["validation"].update(
            onnx_float="passed", onnx_max_abs_error=errors, rtol=2e-3, atol=2e-3
        )
        report.update(
            status="float_checked",
            onnx=str(onnx_path),
            names=str(names_path),
            opset=17 if variant.startswith("26") else 11,
        )
    except Exception as error:
        for stage, value in report["validation"].items():
            if value == "running":
                report["validation"][stage] = "failed"
        report.update(status="export_failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        torch.set_num_threads(original_threads)
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        (root / "export.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    return report


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--test-image", type=Path)
    parser.add_argument("--threads", type=int, default=2)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        result = export_checkpoint(
            weights=args.weights,
            variant=args.variant,
            output_dir=args.output_dir,
            test_image=args.test_image,
            threads=args.threads,
        )
        print(json.dumps(result, indent=2))
        return 0
    except (ValueError, OSError, ImportError, RuntimeError, AssertionError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
