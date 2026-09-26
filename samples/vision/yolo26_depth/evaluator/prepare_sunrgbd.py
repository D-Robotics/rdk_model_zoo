# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Prepare all three source evaluation protocols with explicit record identity."""

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
from samples._shared.assets import sha256_file
from samples.vision.yolo26_depth.conversion.prepare_calibration import prepare_tensor

PROTOCOLS = ("deployment_letterbox", "deployment_scale_fill", "ultralytics_validator")


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
        image = cv2.resize(
            image, (stage1_width, stage1_height), interpolation=cv2.INTER_LINEAR
        )
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


def proportional_allocations(groups, count):
    """Source stratified allocation, including count smaller than sensor count."""
    total = sum(len(v) for v in groups.values())
    if type(count) is not int or not 0 <= count <= total:
        raise ValueError("Screen count must be between zero and dataset size")
    if not total or count == 0:
        return {key: 0 for key in groups}
    raw = {key: count * len(values) / total for key, values in groups.items()}
    allocations = {key: math.floor(value) for key, value in raw.items()}
    minimum = 1 if count >= sum(bool(v) for v in groups.values()) else 0
    for key in groups:
        if groups[key]:
            allocations[key] = max(allocations[key], minimum)
    while sum(allocations.values()) < count:
        eligible = [key for key in groups if allocations[key] < len(groups[key])]
        key = max(eligible, key=lambda k: (raw[k] - allocations[k], len(groups[k]), k))
        allocations[key] += 1
    while sum(allocations.values()) > count:
        eligible = [key for key in groups if allocations[key] > minimum]
        key = min(eligible, key=lambda k: (raw[k] - allocations[k], -len(groups[k]), k))
        allocations[key] -= 1
    return allocations


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-root", type=Path, required=True)
    p.add_argument("--source-manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--size", type=int, choices=(768,), default=768)
    p.add_argument("--screen-count", type=int, default=20)
    p.add_argument("--screen-seed", type=int, default=20260726)
    p.add_argument(
        "--protocol",
        choices=PROTOCOLS,
        action="append",
        help="Repeat to select protocols; default all three",
    )
    args = p.parse_args(argv)
    if args.screen_count < 0:
        raise ValueError("screen-count must be nonnegative")
    if args.output.exists():
        raise FileExistsError(args.output)
    protocols = tuple(dict.fromkeys(args.protocol or PROTOCOLS))
    source = json.loads(args.source_manifest.read_text())["records"]
    if not source:
        raise ValueError("Source manifest has no records")
    groups = defaultdict(list)
    ids = []
    for position, record in enumerate(source):
        index = record.get("index", position)
        if type(index) is not int or index < 0 or index in ids:
            raise ValueError(
                "Source record indices must be unique nonnegative integers"
            )
        ids.append(index)
        groups[record.get("sensor", "unknown")].append(index)
    allocation = proportional_allocations(groups, min(args.screen_count, len(source)))
    rng = random.Random(args.screen_seed)
    screen = sorted(
        i
        for sensor in sorted(groups)
        for i in rng.sample(sorted(groups[sensor]), allocation[sensor])
    )
    args.output.mkdir(parents=True, exist_ok=False)
    result = []
    for index, record in zip(ids, source):
        image_path = args.source_root / record["image"]
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Cannot decode {image_path}")
        if record.get("image_hw", list(image.shape[:2])) != list(image.shape[:2]):
            raise ValueError(
                f"Image geometry differs from source manifest: {image_path}"
            )
        item = {
            "index": index,
            "sample": Path(record["image"]).stem,
            "sensor": record.get("sensor", "unknown"),
            "image": record["image"],
            "depth_m": record.get("depth_m"),
            "original_hw": list(image.shape[:2]),
            "image_sha256": sha256_file(image_path),
            "screen": index in screen,
        }
        for protocol in protocols:
            if protocol == "deployment_letterbox":
                tensor, geometry = prepare_tensor(image, "x5", "nv12")
            elif protocol == "deployment_scale_fill":
                tensor, geometry = prepare_tensor(image, "s100", "lite")
            else:
                transformed, geometry = validator_stretch(image, args.size)
                tensor = to_rgb_chw(transformed)
            floating = protocol == "deployment_scale_fill"
            directory = (
                args.output / protocol / ("featuremap" if floating else "rgb_chw_u8")
            )
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / f'{index:04d}{".npy" if floating else ".bin"}'
            if floating:
                np.save(path, tensor, allow_pickle=False)
            else:
                tensor.tofile(path)
            kind = "npy" if floating else "bin"
            item[protocol] = {
                kind: path.relative_to(args.output).as_posix(),
                kind + "_sha256": sha256_file(path),
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                **geometry,
            }
        result.append(item)
    manifest = {
        "schema_version": "2.0",
        "source_root": str(args.source_root.resolve()),
        "source_manifest": str(args.source_manifest.resolve()),
        "source_manifest_sha256": sha256_file(args.source_manifest),
        "size": args.size,
        "sample_count": len(result),
        "records": result,
        "screen_selection": {
            "count": len(screen),
            "seed": args.screen_seed,
            "method": "proportional stratified by sensor; lower-count largest remainder",
            "allocations": allocation,
            "indices": screen,
        },
        "protocols": {
            key: {
                "representation": (
                    "RGB float32 NCHW /255 .npy"
                    if key == "deployment_scale_fill"
                    else "RGB uint8 CHW .bin; normalize in model/adapter"
                ),
                "boundary": (
                    "raw logit"
                    if key == "deployment_scale_fill"
                    else "calibrated log-depth"
                ),
            }
            for key in protocols
        },
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            {
                "manifest": str(args.output / "manifest.json"),
                "records": len(result),
                "protocols": protocols,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
