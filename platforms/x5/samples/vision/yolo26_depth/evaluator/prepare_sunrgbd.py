"""Prepare deterministic SUN RGB-D inputs and metadata for evaluation."""

import argparse
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


def sha256(path: Path) -> str:
    """Calculate a SHA256 digest for a prepared artifact.

    Args:
        path: File to hash.

    Returns:
        Lowercase hexadecimal SHA256 digest.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def proportional_allocations(groups: dict[str, list[int]], count: int) -> dict[str, int]:
    """Allocate a deterministic record count across dataset groups.

    Args:
        groups: Mapping from group names to record identifiers.
        count: Total number of records to select.

    Returns:
        Per-group selection counts.
    """
    total = sum(len(items) for items in groups.values())
    raw = {key: count * len(items) / total for key, items in groups.items()}
    allocations = {key: math.floor(value) for key, value in raw.items()}
    for key in allocations:
        if allocations[key] == 0 and groups[key]:
            allocations[key] = 1
    while sum(allocations.values()) < count:
        key = max(groups, key=lambda item: (raw[item] - allocations[item], len(groups[item]), item))
        allocations[key] += 1
    while sum(allocations.values()) > count:
        candidates = [key for key, value in allocations.items() if value > 1]
        key = min(candidates, key=lambda item: (raw[item] - allocations[item], -len(groups[item]), item))
        allocations[key] -= 1
    return allocations


def letterbox(image: np.ndarray, size: int) -> tuple[np.ndarray, dict]:
    """Resize one image with deployment-compatible letterbox padding.

    Args:
        image: Source BGR image.
        size: Square target resolution.

    Returns:
        Padded image and preprocessing metadata.
    """
    height, width = image.shape[:2]
    ratio = min(size / height, size / width)
    resized_width = round(width * ratio)
    resized_height = round(height * ratio)
    if (width, height) != (resized_width, resized_height):
        image = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    pad_width = size - resized_width
    pad_height = size - resized_height
    top = round(pad_height / 2 - 0.1)
    bottom = round(pad_height / 2 + 0.1)
    left = round(pad_width / 2 - 0.1)
    right = round(pad_width / 2 + 0.1)
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
        "ratio": ratio,
        "resized_hw": [resized_height, resized_width],
        "padding_tblr": [top, bottom, left, right],
    }


def validator_stretch(image: np.ndarray, size: int) -> tuple[np.ndarray, dict]:
    """Resize one image with the Ultralytics validator stretch policy.

    Args:
        image: Source BGR image.
        size: Square target resolution.

    Returns:
        Resized image and preprocessing metadata.
    """
    height, width = image.shape[:2]
    ratio = size / max(height, width)
    stage1_height, stage1_width = height, width
    if ratio != 1:
        stage1_width = min(math.ceil(width * ratio), size)
        stage1_height = min(math.ceil(height * ratio), size)
        image = cv2.resize(image, (stage1_width, stage1_height), interpolation=cv2.INTER_LINEAR)
    if image.shape[:2] != (size, size):
        image = cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)
    return image, {
        "long_side_ratio": ratio,
        "stage1_hw": [stage1_height, stage1_width],
        "output_hw": [size, size],
    }


def to_rgb_chw(image: np.ndarray) -> np.ndarray:
    """Convert BGR HWC input to contiguous RGB CHW format.

    Args:
        image: Source BGR HWC image.

    Returns:
        Contiguous uint8 RGB CHW tensor.
    """
    return np.ascontiguousarray(image[:, :, ::-1].transpose(2, 0, 1), dtype=np.uint8)


def main() -> None:
    """Prepare a deterministic SUN RGB-D evaluation subset."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--source-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--size", type=int, default=768)
    parser.add_argument("--screen-count", type=int, default=20)
    parser.add_argument("--screen-seed", type=int, default=20260726)
    args = parser.parse_args()

    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"output is not empty: {args.output}")
    deployment_dir = args.output / "deployment_letterbox" / "rgb_chw_u8"
    validator_dir = args.output / "ultralytics_validator" / "rgb_chw_u8"
    deployment_dir.mkdir(parents=True, exist_ok=True)
    validator_dir.mkdir(parents=True, exist_ok=True)

    source_manifest = json.loads(args.source_manifest.read_text(encoding="utf-8"))
    source_records = source_manifest["records"]
    groups: dict[str, list[int]] = defaultdict(list)
    for record in source_records:
        groups[record["sensor"]].append(int(record["index"]))
    allocations = proportional_allocations(groups, min(args.screen_count, len(source_records)))
    rng = random.Random(args.screen_seed)
    screen_indices = []
    for sensor in sorted(groups):
        screen_indices.extend(rng.sample(sorted(groups[sensor]), allocations[sensor]))
    screen_indices.sort()

    records = []
    for record in source_records:
        index = int(record["index"])
        image_path = args.source_root / record["image"]
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"failed to decode {image_path}")
        original_hw = list(image.shape[:2])
        if original_hw != record["image_hw"]:
            raise ValueError(f"shape mismatch for {image_path}: {original_hw} != {record['image_hw']}")

        deployment_image, deployment_geometry = letterbox(image.copy(), args.size)
        validator_image, validator_geometry = validator_stretch(image.copy(), args.size)
        deployment_tensor = to_rgb_chw(deployment_image)
        validator_tensor = to_rgb_chw(validator_image)
        deployment_bin = deployment_dir / f"{index:04d}.bin"
        validator_bin = validator_dir / f"{index:04d}.bin"
        deployment_tensor.tofile(deployment_bin)
        validator_tensor.tofile(validator_bin)

        records.append(
            {
                "index": index,
                "sample": Path(record["image"]).stem,
                "sensor": record["sensor"],
                "image": record["image"],
                "depth_m": record.get("depth_m"),
                "original_hw": original_hw,
                "image_sha256": sha256(image_path),
                "screen": index in screen_indices,
                "deployment_letterbox": {
                    "bin": deployment_bin.relative_to(args.output).as_posix(),
                    "bin_sha256": sha256(deployment_bin),
                    "shape": list(deployment_tensor.shape),
                    **deployment_geometry,
                },
                "ultralytics_validator": {
                    "bin": validator_bin.relative_to(args.output).as_posix(),
                    "bin_sha256": sha256(validator_bin),
                    "shape": list(validator_tensor.shape),
                    **validator_geometry,
                },
            }
        )

    manifest = {
        "schema_version": "1.0",
        "source_root": str(args.source_root.resolve()),
        "source_manifest": str(args.source_manifest.resolve()),
        "size": args.size,
        "sample_count": len(records),
        "screen_selection": {
            "count": len(screen_indices),
            "seed": args.screen_seed,
            "method": "proportional stratified sample by sensor",
            "allocations": allocations,
            "indices": screen_indices,
        },
        "input_contract": {
            "shape": [1, 3, args.size, args.size],
            "sample_shape": [3, args.size, args.size],
            "dtype": "uint8",
            "layout": "CHW",
            "color": "RGB",
            "normalization": "none in files; model or mapper applies scale 1/255",
        },
        "protocols": {
            "deployment_letterbox": {
                "preprocess": "fixed-square Ultralytics predictor letterbox with padding 114",
                "postprocess": "exp(log_depth), bilinear resize to input size with align_corners=False, crop padding, bilinear resize to native GT",
            },
            "ultralytics_validator": {
                "preprocess": "DepthDataset long-side resize followed by scale_fill square resize",
                "ground_truth": "same two-stage geometry with nearest-neighbor interpolation",
                "postprocess": "exp(log_depth), bilinear resize to input size with align_corners=False",
            },
        },
        "records": records,
    }
    manifest_path = args.output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (args.output / "screen_indices.txt").write_text(
        "\n".join(str(index) for index in screen_indices) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "samples": len(records),
                "screen_indices": screen_indices,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
