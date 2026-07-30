"""YOLO26n Depth X5 conversion utility used by this Model Zoo sample."""

import argparse
import hashlib
import json
import random
from pathlib import Path

import cv2
import numpy as np


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def sha256(path: Path) -> str:
    """Calculate a SHA256 digest for a calibration artifact.

    Args:
        path: File to hash.

    Returns:
        Lowercase hexadecimal SHA256 digest.
    """
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def letterbox(image: np.ndarray, size: int) -> tuple[np.ndarray, dict]:
    """Resize one calibration image with 114-value letterbox padding.

    Args:
        image: Source BGR image.
        size: Square calibration resolution in pixels.

    Returns:
        Padded image and a dictionary describing resize geometry.
    """
    height, width = image.shape[:2]
    ratio = min(size / height, size / width)
    resized_width = round(width * ratio)
    resized_height = round(height * ratio)
    pad_width = size - resized_width
    pad_height = size - resized_height
    half_width = pad_width / 2
    half_height = pad_height / 2

    if (width, height) != (resized_width, resized_height):
        image = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    top = round(half_height - 0.1)
    bottom = round(half_height + 0.1)
    left = round(half_width - 0.1)
    right = round(half_width + 0.1)
    image = cv2.copyMakeBorder(
        image,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    )
    return image, {
        "original_hw": [height, width],
        "ratio": ratio,
        "resized_hw": [resized_height, resized_width],
        "padding_tblr": [top, bottom, left, right],
    }


def main() -> None:
    """Generate deterministic RGB CHW calibration tensors."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
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
    selected = random.Random(args.seed).sample(candidates, args.count)

    records = []
    invalid = []
    for index, source in enumerate(selected):
        bgr = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if bgr is None:
            invalid.append(str(source))
            continue
        padded, geometry = letterbox(bgr, args.size)
        rgb_chw = np.ascontiguousarray(padded[:, :, ::-1].transpose(2, 0, 1), dtype=np.uint8)
        output = args.output / f"{index:04d}.bin"
        rgb_chw.tofile(output)
        records.append(
            {
                "index": index,
                "source": str(source.relative_to(args.images)),
                "source_sha256": sha256(source),
                "output": output.name,
                "output_sha256": sha256(output),
                "shape": list(rgb_chw.shape),
                "dtype": str(rgb_chw.dtype),
                "min": int(rgb_chw.min()),
                "max": int(rgb_chw.max()),
                "mean": float(rgb_chw.mean()),
                "non_finite": 0,
                **geometry,
            }
        )

    if invalid or len(records) != args.count:
        raise RuntimeError(f"invalid images: {invalid}; produced {len(records)} records")

    manifest = {
        "schema_version": "1.0",
        "source_root": str(args.images),
        "selection": {"method": "seeded random sample", "seed": args.seed, "count": args.count},
        "input_contract": {
            "name": "images",
            "shape": [1, 3, args.size, args.size],
            "sample_shape": [3, args.size, args.size],
            "dtype": "uint8",
            "layout": "CHW",
            "color": "RGB",
            "normalization": "none in files; mapper data_scale=1/255",
        },
        "records": records,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    means = np.array([record["mean"] for record in records], dtype=np.float64)
    report = "\n".join(
        [
            "# Calibration Preprocess Report",
            "",
            f"- Samples: {len(records)}",
            f"- Selection seed: {args.seed}",
            f"- Output contract: RGB CHW uint8, 3x{args.size}x{args.size}",
            "- Geometry: aspect-ratio preserving resize plus symmetric padding 114",
            "- Normalization: deferred to OE Mapper with data_scale=0.003921568627451",
            f"- Per-image mean range: {means.min():.6f} to {means.max():.6f}",
            f"- Per-image mean average: {means.mean():.6f}",
            "- NaN/Inf count: 0 (uint8)",
            "",
        ]
    )
    args.report.write_text(report, encoding="utf-8")
    print(json.dumps({"samples": len(records), "manifest": str(args.manifest)}, indent=2))


if __name__ == "__main__":
    main()
