#!/usr/bin/env python3
# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Export the pinned public Small checkpoint and verify real/random host parity.

The upstream checkout is external and must match SOURCE_COMMIT. No dependency
installation, source mutation, download, or compiler invocation occurs here.
"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

SOURCE_COMMIT = "a561b849ebae10a6f5ef49e26c83cbbcd36c71bf"
CHECKPOINT_SHA256 = "715fade13be8f229f8a70cc02066f656f2423a59effd0579197bbf57860e1378"
HEIGHT, WIDTH = 518, 686


def digest(path):
    """Hash one source or generated file."""
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def preprocess(image):
    """BGR uint8 to fixed-size RGB/ImageNet F32, matching the Small binding."""
    import cv2
    import numpy as np

    if image is None or image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("Expected a decoded BGR uint8 image")
    # Upstream converts to float64 before cubic resize and normalization.
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
    rgb = cv2.resize(rgb, (WIDTH, HEIGHT), interpolation=cv2.INTER_CUBIC)
    normalized = (rgb - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    return np.ascontiguousarray(normalized.transpose(2, 0, 1)[None], dtype=np.float32)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args(argv)
    commit = subprocess.check_output(["git", "-C", str(args.source), "rev-parse", "HEAD"], text=True).strip()
    if commit != SOURCE_COMMIT:
        raise ValueError(f"Expected source {SOURCE_COMMIT}, found {commit}")
    if subprocess.check_output(["git", "-C", str(args.source), "status", "--porcelain"], text=True).strip():
        raise ValueError("The pinned upstream source must be clean")
    if digest(args.checkpoint) != CHECKPOINT_SHA256:
        raise ValueError("Official Small checkpoint SHA-256 mismatch")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    import cv2
    import numpy as np
    import onnx
    import onnxruntime as ort
    import torch

    sys.path.insert(0, str(args.source.resolve()))
    from depth_anything_v2.dpt import DepthAnythingV2

    torch.set_num_threads(args.threads)
    torch.manual_seed(42)
    model = DepthAnythingV2(encoder="vits", features=64, out_channels=[48, 96, 192, 384])
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True), strict=True)
    model.eval()
    real = preprocess(cv2.imread(str(args.image)))
    random = np.random.default_rng(42).standard_normal((1, 3, HEIGHT, WIDTH)).astype(np.float32)
    model_path = args.output_dir / "depth_anything_v2_small_518x686.onnx"
    with torch.inference_mode():
        torch.onnx.export(model, torch.from_numpy(real), str(model_path),
                          input_names=["images"], output_names=["depth"],
                          opset_version=17, dynamo=False, do_constant_folding=True)
    graph = onnx.load(model_path)
    onnx.checker.check_model(graph)
    options = ort.SessionOptions()
    options.intra_op_num_threads = args.threads
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(model_path), options, providers=["CPUExecutionProvider"])
    comparisons = []
    for name, value in (("real", real), ("random_seed42", random)):
        with torch.inference_mode():
            expected = model(torch.from_numpy(value)).numpy()
        observed = session.run(["depth"], {"images": value})[0]
        if expected.shape != (1, HEIGHT, WIDTH) or observed.shape != expected.shape:
            raise AssertionError("Unexpected depth output shape")
        if not np.isfinite(expected).all() or not np.isfinite(observed).all():
            raise AssertionError("Depth outputs must be finite")
        np.testing.assert_allclose(observed, expected, rtol=1e-3, atol=1e-3)
        delta = np.abs(observed.astype(np.float64) - expected)
        comparisons.append({"input": name, "max_abs_error": float(delta.max()),
                            "mean_abs_error": float(delta.mean()), "rtol": 1e-3, "atol": 1e-3,
                            "finite": True, "passed": True})
        np.save(args.output_dir / f"{name}_input.npy", value)
        np.save(args.output_dir / f"{name}_torch.npy", expected)
        np.save(args.output_dir / f"{name}_ort.npy", observed)
    report = {
        "status": "passed", "source_commit": commit,
        "checkpoint_sha256": CHECKPOINT_SHA256, "checkpoint_size_bytes": args.checkpoint.stat().st_size,
        "onnx_sha256": digest(model_path), "onnx_size_bytes": model_path.stat().st_size,
        "exporter_sha256": digest(__file__), "image_sha256": digest(args.image),
        "model": "DepthAnythingV2(encoder=vits,features=64,out_channels=[48,96,192,384])",
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "source_modifications": [], "graph_refinements": [], "opset": 17,
        "input": {"name": "images", "shape": [1, 3, HEIGHT, WIDTH], "dtype": "float32"},
        "output": {"name": "depth", "shape": [1, HEIGHT, WIDTH], "dtype": "float32"},
        "preprocess": "BGR->RGB float64/255; cubic stretch to 518x686; ImageNet mean/std; F32 NCHW",
        "versions": {"torch": torch.__version__, "onnx": onnx.__version__,
                     "onnxruntime": ort.__version__, "numpy": np.__version__, "opencv": cv2.__version__},
        "comparisons": comparisons,
    }
    (args.output_dir / "host_parity.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
