"""Prepare SUN RGB-D featuremap tensors for the deployed YOLO26 Depth lite model."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import cv2
import numpy as np


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def preprocess(image: np.ndarray, size: int) -> np.ndarray:
    """Scale-fill BGR to RGB /255 float32 NCHW, matching board inference."""
    resized = cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)
    return np.ascontiguousarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)[None], dtype=np.float32) / 255.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--source-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--size", type=int, default=768)
    parser.add_argument("--screen-count", type=int, default=20)
    parser.add_argument("--screen-seed", type=int, default=20260726)
    args = parser.parse_args()
    records = json.loads(args.source_manifest.read_text(encoding="utf-8"))["records"]
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"output is not empty: {args.output}")
    tensor_dir = args.output / "deployment_scale_fill" / "featuremap"
    tensor_dir.mkdir(parents=True)
    screen = set(random.Random(args.screen_seed).sample(range(len(records)), min(args.screen_count, len(records))))
    result = []
    for index, source in enumerate(records):
        image_path = args.source_root / source["image"]
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"cannot decode {image_path}")
        tensor = preprocess(image, args.size)
        output = tensor_dir / f"{index:04d}.npy"
        np.save(output, tensor)
        result.append({"index": index, "sample": Path(source["image"]).stem, "sensor": source.get("sensor", "unknown"), "image": source["image"], "depth_m": source.get("depth_m"), "original_hw": list(image.shape[:2]), "image_sha256": digest(image_path), "screen": index in screen, "deployment_scale_fill": {"npy": output.relative_to(args.output).as_posix(), "npy_sha256": digest(output), "shape": list(tensor.shape)}})
    manifest = {"schema_version": "1.0", "source_root": str(args.source_root.resolve()), "source_manifest": str(args.source_manifest.resolve()), "size": args.size, "sample_count": len(result), "screen_selection": {"count": len(screen), "seed": args.screen_seed, "method": "seeded sample", "indices": sorted(screen)}, "input_contract": {"shape": [1, 3, args.size, args.size], "dtype": "float32", "layout": "NCHW", "color": "RGB", "resize": "scale-fill cv2.INTER_LINEAR", "normalization": "/255"}, "protocols": {"deployment_scale_fill": {"postprocess": "clip(raw,-4,5)*cal_a+cal_b; exp; resize to native ground truth"}}, "records": result}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
