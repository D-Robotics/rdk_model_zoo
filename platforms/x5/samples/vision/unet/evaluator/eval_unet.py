#!/usr/bin/env python3
# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0

"""Evaluate UNet ResNet checkpoints, ONNX models, or X5 binaries on VOC pairs."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image

NUM_CLASSES = 21
INPUT_SIZE = 512
IGNORE_INDEX = 255
BACKBONES = ("resnet18", "resnet34", "resnet50", "resnet101", "resnet152")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_pairs(path: Path) -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        fields = line.split("\t")
        if len(fields) != 2:
            raise ValueError(f"{path}:{line_number}: expected image<TAB>mask")
        image, mask = (Path(field).expanduser().resolve() for field in fields)
        if not image.is_file() or not mask.is_file():
            raise FileNotFoundError(f"missing pair at {path}:{line_number}")
        pairs.append((image, mask))
    if not pairs:
        raise ValueError(f"manifest has no samples: {path}")
    return pairs


def load_sample(image_path: Path, mask_path: Path) -> tuple[np.ndarray, np.ndarray]:
    size = (INPUT_SIZE, INPUT_SIZE)
    with Image.open(image_path) as source:
        image = source.convert("RGB").resize(
            size,
            resample=Image.Resampling.BILINEAR,
        )
    with Image.open(mask_path) as source:
        if source.mode not in {"L", "P"}:
            raise ValueError(
                f"mask must store class indices in L or P mode: {mask_path}"
            )
        mask = source.resize(size, resample=Image.Resampling.NEAREST)

    image_array = np.asarray(image, dtype=np.uint8).copy()
    mask_array = np.asarray(mask, dtype=np.uint8).astype(np.int64)
    valid = (mask_array < NUM_CLASSES) | (mask_array == IGNORE_INDEX)
    if not bool(valid.all()):
        raise ValueError(f"mask contains invalid class ids: {mask_path}")
    return image_array, mask_array


def rgb_to_nchw(image: np.ndarray) -> np.ndarray:
    tensor = image.astype(np.float32) / 255.0
    return np.ascontiguousarray(tensor.transpose(2, 0, 1)[None])


def rgb_to_nv12(image: np.ndarray) -> np.ndarray:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("the X5 backend requires OpenCV (cv2)") from exc

    height, width = image.shape[:2]
    if height % 2 or width % 2:
        raise ValueError("NV12 input dimensions must be even")
    i420 = cv2.cvtColor(image, cv2.COLOR_RGB2YUV_I420).reshape(-1)
    y_size = height * width
    chroma_size = y_size // 4
    y_plane = i420[:y_size].reshape(height, width)
    u_plane = i420[y_size : y_size + chroma_size].reshape(
        height // 2,
        width // 2,
    )
    v_plane = i420[y_size + chroma_size :].reshape(
        height // 2,
        width // 2,
    )
    uv_plane = np.empty((height // 2, width), dtype=np.uint8)
    uv_plane[:, 0::2] = u_plane
    uv_plane[:, 1::2] = v_plane
    return np.ascontiguousarray(np.vstack((y_plane, uv_plane)))


def extract_state_dict(checkpoint: object) -> dict[str, Any]:
    if not isinstance(checkpoint, dict):
        raise TypeError("checkpoint must contain a state dictionary")
    if "model_state_dict" in checkpoint:
        state = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint:
        state = checkpoint["state_dict"]
    else:
        state = checkpoint
    if not isinstance(state, dict) or not state:
        raise TypeError("checkpoint state dictionary is empty or invalid")
    return state


def make_pytorch_runner(
    model_path: Path,
    backbone: str | None,
) -> tuple[Callable[[np.ndarray], np.ndarray], dict[str, object]]:
    if backbone is None:
        raise ValueError("--backbone is required for a PyTorch checkpoint")
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("the PyTorch backend requires torch") from exc

    export_root = Path(__file__).resolve().parents[1] / "conversion" / "onnx_export"
    sys.path.insert(0, str(export_root))
    from model import UNet

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    model = UNet(backbone=backbone)
    result = model.load_state_dict(extract_state_dict(checkpoint), strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(f"strict checkpoint load failed: {result}")
    model.eval().to(device)

    def run(image: np.ndarray) -> np.ndarray:
        tensor = torch.from_numpy(rgb_to_nchw(image)).to(device)
        with torch.inference_mode():
            output = model(tensor)
        return output.detach().cpu().numpy()

    metadata = {
        "backend": "pytorch",
        "backbone": backbone,
        "device": str(device),
        "runtime_version": torch.__version__,
    }
    return run, metadata


def make_onnx_runner(
    model_path: Path,
) -> tuple[Callable[[np.ndarray], np.ndarray], dict[str, object]]:
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError("the ONNX backend requires onnxruntime") from exc

    session = ort.InferenceSession(
        str(model_path),
        providers=["CPUExecutionProvider"],
    )
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    if len(inputs) != 1 or len(outputs) != 1:
        raise ValueError("UNet evaluator requires exactly one input and one output")
    input_name = inputs[0].name
    output_name = outputs[0].name

    def run(image: np.ndarray) -> np.ndarray:
        return np.asarray(
            session.run([output_name], {input_name: rgb_to_nchw(image)})[0]
        )

    metadata = {
        "backend": "onnx",
        "input_name": input_name,
        "input_shape": inputs[0].shape,
        "output_name": output_name,
        "output_shape": outputs[0].shape,
        "runtime_version": ort.__version__,
    }
    return run, metadata


def parse_rdk_version(version_text: str) -> tuple[int, int, int]:
    match = re.search(r"(?<!\d)(\d+)\.(\d+)\.(\d+)(?!\d)", version_text)
    if match is None:
        raise RuntimeError("cannot parse an RDK version from /etc/version")
    return tuple(int(value) for value in match.groups())


def require_x5_runtime_environment() -> str:
    if platform.machine() not in {"aarch64", "arm64"}:
        raise RuntimeError("the X5 .bin backend must run on an aarch64 RDK X5")
    version_path = Path("/etc/version")
    if not version_path.is_file():
        raise RuntimeError("/etc/version is missing; cannot verify the X5 runtime")
    version_text = version_path.read_text(encoding="utf-8").strip()
    if parse_rdk_version(version_text) < (3, 5, 0):
        raise RuntimeError("X5 hbm_runtime evaluation requires RDK OS >= 3.5.0")
    return version_text


def dequantize_output(array: np.ndarray, quant: object) -> np.ndarray:
    quant_type = getattr(getattr(quant, "quant_type", None), "name", "NONE")
    if quant_type != "SCALE":
        return array
    scale = np.asarray(getattr(quant, "scale"), dtype=np.float32)
    zero_point = np.asarray(getattr(quant, "zero_point"), dtype=np.float32)
    if scale.size == 1:
        return (array.astype(np.float32) - float(zero_point.reshape(-1)[0])) * float(
            scale.reshape(-1)[0]
        )
    axis = int(getattr(quant, "axis"))
    if axis < 0:
        axis += array.ndim
    shape = [1] * array.ndim
    shape[axis] = scale.size
    return (array.astype(np.float32) - zero_point.reshape(shape)) * scale.reshape(shape)


def make_x5_runner(
    model_path: Path,
) -> tuple[Callable[[np.ndarray], np.ndarray], dict[str, object]]:
    os_version = require_x5_runtime_environment()
    try:
        from hbm_runtime import HB_HBMRuntime
    except ImportError as exc:
        raise RuntimeError(
            "hbm_runtime is unavailable; use the X5 package shipped with RDK OS, "
            "not a PyPI package for another platform"
        ) from exc

    runtime = HB_HBMRuntime(str(model_path))
    if runtime.model_count != 1:
        raise ValueError("UNet evaluator requires a .bin containing one model")
    model_name = runtime.model_names[0]
    input_names = runtime.input_names[model_name]
    output_names = runtime.output_names[model_name]
    if len(input_names) != 1 or len(output_names) != 1:
        raise ValueError("UNet evaluator requires exactly one input and one output")
    input_name = input_names[0]
    output_name = output_names[0]
    input_dtype = runtime.input_dtypes[model_name][input_name].name
    if input_dtype != "NV12":
        raise ValueError(f"expected packed NV12 X5 input, got {input_dtype}")
    output_quant = runtime.output_quants[model_name][output_name]

    def run(image: np.ndarray) -> np.ndarray:
        results = runtime.run({input_name: rgb_to_nv12(image)})
        output = np.asarray(results[model_name][output_name])
        return dequantize_output(output, output_quant)

    metadata = {
        "backend": "x5",
        "os_version": os_version,
        "runtime_version": HB_HBMRuntime.version,
        "model_name": model_name,
        "input_name": input_name,
        "input_shape": runtime.input_shapes[model_name][input_name],
        "input_dtype": input_dtype,
        "output_name": output_name,
        "output_shape": runtime.output_shapes[model_name][output_name],
        "output_dtype": runtime.output_dtypes[model_name][output_name].name,
    }
    return run, metadata


def resolve_backend(model_path: Path, requested: str) -> str:
    if requested != "auto":
        return requested
    suffix = model_path.suffix.lower()
    backends = {
        ".pth": "pytorch",
        ".pt": "pytorch",
        ".onnx": "onnx",
        ".bin": "x5",
    }
    if suffix not in backends:
        raise ValueError(f"cannot infer evaluator backend from {model_path.name}")
    return backends[suffix]


def make_runner(
    backend: str,
    model_path: Path,
    backbone: str | None,
) -> tuple[Callable[[np.ndarray], np.ndarray], dict[str, object]]:
    if backend == "pytorch":
        return make_pytorch_runner(model_path, backbone)
    if backend == "onnx":
        return make_onnx_runner(model_path)
    if backend == "x5":
        return make_x5_runner(model_path)
    raise ValueError(f"unsupported evaluator backend: {backend}")


def prediction_from_output(output: np.ndarray) -> np.ndarray:
    array = np.asarray(output)
    if array.ndim == 4:
        if array.shape[0] != 1:
            raise ValueError(f"expected output batch 1, got shape {array.shape}")
        array = array[0]
    if array.ndim == 3 and array.shape[0] == NUM_CLASSES:
        return array.argmax(axis=0).astype(np.int64)
    if array.ndim == 3 and array.shape[-1] == NUM_CLASSES:
        return array.argmax(axis=-1).astype(np.int64)
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[..., 0]
    if array.ndim == 2:
        prediction = array.astype(np.int64)
        if prediction.size and (
            int(prediction.min()) < 0 or int(prediction.max()) >= NUM_CLASSES
        ):
            raise ValueError("class-index output contains values outside [0, 20]")
        return prediction
    raise ValueError(f"unsupported UNet output shape: {array.shape}")


def update_confusion(
    confusion: np.ndarray,
    prediction: np.ndarray,
    target: np.ndarray,
) -> None:
    if prediction.shape != target.shape:
        raise ValueError(
            f"prediction/target shape mismatch: {prediction.shape} != {target.shape}"
        )
    valid = target != IGNORE_INDEX
    encoded = target[valid] * NUM_CLASSES + prediction[valid]
    confusion += np.bincount(
        encoded,
        minlength=NUM_CLASSES * NUM_CLASSES,
    ).reshape(NUM_CLASSES, NUM_CLASSES)


def metrics_from_confusion(confusion: np.ndarray) -> dict[str, object]:
    matrix = confusion.astype(np.float64)
    intersection = np.diag(matrix)
    union = matrix.sum(axis=1) + matrix.sum(axis=0) - intersection
    valid = union > 0
    iou = np.zeros(NUM_CLASSES, dtype=np.float64)
    iou[valid] = intersection[valid] / union[valid]
    pixel_total = matrix.sum()
    pixel_accuracy = float(intersection.sum() / pixel_total) if pixel_total else 0.0
    return {
        "miou": float(iou[valid].mean()) if bool(valid.any()) else 0.0,
        "pixel_accuracy": pixel_accuracy,
        "class_iou": iou.tolist(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help=".pth, .onnx, or X5 .bin model",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="image<TAB>mask VOC pair list",
    )
    parser.add_argument(
        "--report",
        type=Path,
        required=True,
        help="new JSON report path",
    )
    parser.add_argument(
        "--backend",
        choices=("auto", "pytorch", "onnx", "x5"),
        default="auto",
        help="inference backend; auto selects it from the model suffix",
    )
    parser.add_argument(
        "--backbone",
        choices=BACKBONES,
        help="required for .pth checkpoints",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="evaluate only the first N samples",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=50,
        help="print progress every N samples",
    )
    parser.add_argument(
        "--min-miou",
        type=float,
        default=0.0,
        help="return 2 below this mIoU",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")
    if args.progress_every <= 0:
        raise ValueError("--progress-every must be positive")
    if not 0.0 <= args.min_miou <= 1.0:
        raise ValueError("--min-miou must be between 0 and 1")
    model_path = args.model.expanduser().resolve()
    manifest_path = args.manifest.expanduser().resolve()
    report_path = args.report.expanduser().resolve()
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    if report_path.exists():
        raise FileExistsError(f"refusing to overwrite report: {report_path}")

    backend = resolve_backend(model_path, args.backend)
    runner, runtime = make_runner(backend, model_path, args.backbone)
    pairs = read_pairs(manifest_path)
    if args.limit is not None:
        pairs = pairs[: args.limit]

    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    start_time = time.time()
    for index, (image_path, mask_path) in enumerate(pairs, start=1):
        image, target = load_sample(image_path, mask_path)
        prediction = prediction_from_output(runner(image))
        update_confusion(confusion, prediction, target)
        if index % args.progress_every == 0 or index == len(pairs):
            elapsed = time.time() - start_time
            progress = {
                "processed": index,
                "total": len(pairs),
                "samples_per_second": index / max(elapsed, 1e-9),
            }
            print(json.dumps(progress, ensure_ascii=False), flush=True)

    metrics = metrics_from_confusion(confusion)
    passed = float(metrics["miou"]) >= args.min_miou
    report = {
        "schema_version": "1.0",
        "model": str(model_path),
        "model_sha256": sha256_file(model_path),
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "samples": len(pairs),
        "input_contract": {
            "size": [INPUT_SIZE, INPUT_SIZE],
            "classes": NUM_CLASSES,
            "ignore_index": IGNORE_INDEX,
        },
        "runtime": runtime,
        "metrics": metrics,
        "min_miou": args.min_miou,
        "passed": passed,
        "elapsed_seconds": time.time() - start_time,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
