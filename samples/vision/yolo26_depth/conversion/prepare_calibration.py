"""Prepare deterministic float32 featuremap calibration tensors for lite PTQ."""

import argparse
import hashlib
import json
import random
from pathlib import Path

import cv2
import numpy as np

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def preprocess(image: np.ndarray, size: int) -> np.ndarray:
    """Use the exact scale-fill, RGB /255 contract consumed on board."""
    resized = cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    return np.ascontiguousarray(rgb.transpose(2, 0, 1)[None], dtype=np.float32) / 255.0


def main() -> None:
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
    records = []
    for index, source in enumerate(random.Random(args.seed).sample(candidates, args.count)):
        bgr = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"cannot decode {source}")
        tensor = preprocess(bgr, args.size)
        output = args.output / f"{index:04d}.npy"
        np.save(output, tensor)
        records.append({"index": index, "source": str(source.relative_to(args.images)),
                        "source_sha256": sha256(source), "output": output.name,
                        "output_sha256": sha256(output), "shape": list(tensor.shape), "dtype": str(tensor.dtype)})
    contract = {"name": "images", "shape": [1, 3, args.size, args.size], "dtype": "float32",
                "layout": "NCHW", "color": "RGB", "resize": "scale-fill cv2.INTER_LINEAR", "normalization": "/255"}
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps({"schema_version": "1.0", "source_root": str(args.images), "selection": {"method": "seeded random sample", "seed": args.seed, "count": args.count}, "input_contract": contract, "records": records}, indent=2), encoding="utf-8")
    args.report.write_text("# Calibration Preprocess Report\n\n- Input: featuremap float32 NCHW, RGB, scale-fill to 768×768, `/255`\n", encoding="utf-8")
    print(json.dumps({"samples": len(records), "manifest": str(args.manifest)}, indent=2))


if __name__ == "__main__":
    main()
