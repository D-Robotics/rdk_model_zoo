"""Prepare deterministic float32 calibration tensors for the mixed PTQ profile.

Two calibration contracts match the two compile profiles (see README.md):

- ``lite`` (l/x): scale-fill resize to 768x768, BGR-to-RGB, ``/255``,
  float32 NCHW. Board runtime applies the identical preprocessing.
- ``nv12`` (n/s/m): aspect-preserving letterbox resize with 114-value
  padding, BGR-to-RGB, full-range ``0..255`` float32 NCHW. The ``/255``
  normalization happens inside the graph via ``data_scale``, so calibration
  tensors must stay full-range.

The two contracts are not interchangeable.
"""

import argparse
import hashlib
import json
import random
from pathlib import Path

import cv2
import numpy as np

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of one file."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def preprocess_lite(image: np.ndarray, size: int) -> np.ndarray:
    """Build one lite-contract tensor: scale-fill, RGB, /255, NCHW float32.

    Args:
        image: Source BGR image with shape ``(height, width, 3)``.
        size: Square calibration resolution in pixels.

    Returns:
        A ``(1, 3, size, size)`` float32 tensor in ``[0, 1]``.
    """
    resized = cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    return np.ascontiguousarray(rgb.transpose(2, 0, 1)[None], dtype=np.float32) / 255.0


def preprocess_nv12(image: np.ndarray, size: int) -> tuple[np.ndarray, dict]:
    """Build one nv12-contract tensor: 114-letterbox, RGB, full-range NCHW float32.

    Args:
        image: Source BGR image with shape ``(height, width, 3)``.
        size: Square calibration resolution in pixels.

    Returns:
        A ``(1, 3, size, size)`` float32 tensor in ``[0, 255]`` and the
        letterbox geometry record.
    """
    height, width = image.shape[:2]
    ratio = min(size / height, size / width)
    resized_width = round(width * ratio)
    resized_height = round(height * ratio)
    pad_width = size - resized_width
    pad_height = size - resized_height
    top = round(pad_height / 2 - 0.1)
    bottom = round(pad_height / 2 + 0.1)
    left = round(pad_width / 2 - 0.1)
    right = round(pad_width / 2 + 0.1)
    if (width, height) != (resized_width, resized_height):
        image = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    padded = cv2.copyMakeBorder(
        image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )
    tensor = np.ascontiguousarray(
        padded[:, :, ::-1].transpose(2, 0, 1)[None], dtype=np.float32
    )
    geometry = {
        "original_hw": [int(height), int(width)],
        "ratio": float(ratio),
        "resized_hw": [int(resized_height), int(resized_width)],
        "padding_tblr": [int(top), int(bottom), int(left), int(right)],
    }
    return tensor, geometry


def main() -> None:
    """Parse options and write the calibration tensor set."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--contract", choices=("lite", "nv12"), default="lite",
                        help="lite: scale-fill /255 (l/x); nv12: 114-letterbox full-range (n/s/m)")
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument("--size", type=int, default=768)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"output already exists: {args.output}")
    args.output.mkdir(parents=True)
    candidates = sorted(path for path in args.images.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)
    if len(candidates) < args.count:
        raise ValueError(f"need {args.count} images, found {len(candidates)}")
    records = []
    for index, source in enumerate(random.Random(args.seed).sample(candidates, args.count)):
        bgr = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"cannot decode {source}")
        if args.contract == "lite":
            tensor = preprocess_lite(bgr, args.size)
            record = {"index": index, "source": str(source.relative_to(args.images)),
                      "source_sha256": sha256(source), "output": f"{index:04d}.npy",
                      "shape": list(tensor.shape), "dtype": str(tensor.dtype)}
        else:
            tensor, geometry = preprocess_nv12(bgr, args.size)
            record = {"index": index, "source": str(source.relative_to(args.images)),
                      "source_sha256": sha256(source), "output": f"{index:04d}.npy",
                      "shape": list(tensor.shape), "dtype": str(tensor.dtype),
                      "min": float(tensor.min()), "max": float(tensor.max()),
                      **geometry}
        output = args.output / record["output"]
        np.save(output, tensor)
        record["output_sha256"] = sha256(output)
        records.append(record)
    if args.contract == "lite":
        contract = {"name": "images", "shape": [1, 3, args.size, args.size], "dtype": "float32",
                    "layout": "NCHW", "color": "RGB", "resize": "scale-fill cv2.INTER_LINEAR",
                    "normalization": "/255"}
        report_note = "# Calibration Preprocess Report\n\n- Profile: lite\n- Input: featuremap float32 NCHW, RGB, scale-fill, `/255`\n"
    else:
        contract = {"name": "images", "shape": [1, 3, args.size, args.size], "dtype": "float32",
                    "layout": "NCHW", "color": "RGB",
                    "resize": "letterbox cv2.INTER_LINEAR, 114 padding",
                    "normalization": "none in files; data_scale=1/255 inside the graph"}
        report_note = "# Calibration Preprocess Report\n\n- Profile: nv12\n- Input: float32 NCHW, RGB, 114-letterbox, full-range 0..255 (`data_scale=1/255` applied in-graph)\n"
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps({"schema_version": "1.0", "profile": args.contract,
                                         "source_root": str(args.images),
                                         "selection": {"method": "seeded random sample", "seed": args.seed, "count": args.count},
                                         "input_contract": contract, "records": records}, indent=2), encoding="utf-8")
    args.report.write_text(report_note, encoding="utf-8")
    print(json.dumps({"samples": len(records), "profile": args.contract, "manifest": str(args.manifest)}, indent=2))


if __name__ == "__main__":
    main()
