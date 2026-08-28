"""YOLO26n Depth X5 conversion utility used by this Model Zoo sample."""

import argparse
import hashlib
import io
import json
import math
import random
import shutil
import zipfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

import cv2
import numpy as np


def sha256(path: Path) -> str:
    """Calculate a SHA256 digest for a file.

    Args:
        path: File to hash.

    Returns:
        Lowercase hexadecimal SHA256 digest.
    """
    """Calculate a SHA256 digest for a file.

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


def proportional_allocations(groups: dict[str, list[str]], count: int) -> dict[str, int]:
    """Allocate a deterministic sample count across dataset groups.

    Args:
        groups: Mapping from group name to candidate record identifiers.
        count: Total number of records to select.

    Returns:
        Per-group selection counts whose sum equals ``count``.
    """
    """Allocate a deterministic sample count across dataset groups.

    Args:
        groups: Mapping from group name to candidate record identifiers.
        count: Total number of records to select.

    Returns:
        Per-group selection counts whose sum equals ``count``.
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


def decode_depth(data: bytes) -> np.ndarray:
    """Decode a SUN RGB-D depth PNG payload.

    Args:
        data: Encoded PNG bytes from the dataset archive.

    Returns:
        Decoded depth map as a NumPy array.
    """
    """Decode a SUN RGB-D depth PNG payload.

    Args:
        data: Encoded PNG bytes from the dataset archive.

    Returns:
        Decoded depth map as a NumPy array.
    """
    encoded = np.frombuffer(data, dtype=np.uint8)
    depth_raw = cv2.imdecode(encoded, cv2.IMREAD_ANYDEPTH)
    if depth_raw is None or depth_raw.dtype != np.uint16:
        raise ValueError("failed to decode uint16 SUN RGB-D depth")
    depth_m = (((depth_raw >> 3) | (depth_raw << 13)) / 1000.0).clip(max=10).astype(np.float32)
    return depth_m


def main() -> None:
    """Extract a deterministic SUN RGB-D calibration subset."""
    """Extract a deterministic SUN RGB-D calibration subset."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--split", choices=("train", "val"), required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"output is not empty: {args.output}")
    image_dir = args.output / "images"
    depth_png_dir = args.output / "depth_bfx"
    depth_npy_dir = args.output / "depth_m"
    image_dir.mkdir(parents=True, exist_ok=True)
    depth_png_dir.mkdir(parents=True, exist_ok=True)
    depth_npy_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(args.archive) as zf:
        infos = zf.infolist()
        scenes = set()
        scene_images = defaultdict(list)
        scene_depth = defaultdict(list)
        for info in infos:
            name = info.filename
            path = PurePosixPath(name)
            if path.is_absolute() or ".." in path.parts:
                raise RuntimeError(f"unsafe ZIP entry: {name}")
            if not name.startswith("SUNRGBD/") or info.is_dir():
                continue
            if "/depth_bfx/" in name and name.lower().endswith(".png"):
                scene = name.split("/depth_bfx/", 1)[0]
                scenes.add(scene)
                scene_depth[scene].append(name)
            elif "/image/" in name and name.lower().endswith((".jpg", ".jpeg")):
                scene = name.split("/image/", 1)[0]
                scene_images[scene].append(name)

        scenes = sorted(scene for scene in scenes if scene_images[scene] and scene_depth[scene])
        names = {scene: "_".join(PurePosixPath(scene).relative_to("SUNRGBD").parts) for scene in scenes}
        val_names = set(random.Random(0).sample([names[scene] for scene in scenes], k=1090))
        pool = [scene for scene in scenes if (names[scene] in val_names) == (args.split == "val")]
        by_sensor = defaultdict(list)
        for scene in pool:
            sensor = PurePosixPath(scene).relative_to("SUNRGBD").parts[0]
            by_sensor[sensor].append(scene)
        allocations = proportional_allocations(by_sensor, args.count)
        rng = random.Random(args.seed)
        selected = []
        for sensor in sorted(by_sensor):
            selected.extend(rng.sample(sorted(by_sensor[sensor]), allocations[sensor]))
        selected.sort(key=lambda scene: (PurePosixPath(scene).relative_to("SUNRGBD").parts[0], scene))

        records = []
        for index, scene in enumerate(selected):
            image_entry = sorted(scene_images[scene])[0]
            depth_entry = sorted(scene_depth[scene])[0]
            scene_name = names[scene]
            sensor = PurePosixPath(scene).relative_to("SUNRGBD").parts[0]
            image_suffix = Path(image_entry).suffix.lower()
            stem = f"{index:04d}_{scene_name}"
            image_path = image_dir / f"{stem}{image_suffix}"
            depth_png_path = depth_png_dir / f"{stem}.png"
            depth_npy_path = depth_npy_dir / f"{stem}.npy"
            with zf.open(image_entry) as source, image_path.open("wb") as target:
                shutil.copyfileobj(source, target, length=1024 * 1024)
            depth_bytes = zf.read(depth_entry)
            depth_png_path.write_bytes(depth_bytes)
            depth_m = decode_depth(depth_bytes)
            np.save(depth_npy_path, depth_m)
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"failed to decode {image_path}")
            records.append({
                "index": index,
                "split": args.split,
                "sensor": sensor,
                "scene": scene,
                "scene_name": scene_name,
                "image_zip_entry": image_entry,
                "depth_zip_entry": depth_entry,
                "image": image_path.relative_to(args.output).as_posix(),
                "depth_png": depth_png_path.relative_to(args.output).as_posix(),
                "depth_m": depth_npy_path.relative_to(args.output).as_posix(),
                "image_hw": list(image.shape[:2]),
                "depth_hw": list(depth_m.shape),
                "shape_match": list(image.shape[:2]) == list(depth_m.shape),
                "valid_depth_pixels": int(np.count_nonzero(depth_m > 0)),
                "depth_min_positive": float(depth_m[depth_m > 0].min()) if np.any(depth_m > 0) else None,
                "depth_max": float(depth_m.max()),
                "image_sha256": sha256(image_path),
                "depth_png_sha256": sha256(depth_png_path),
                "depth_npy_sha256": sha256(depth_npy_path),
            })

    manifest = {
        "schema_version": "1.0",
        "archive": str(args.archive.resolve()),
        "archive_bytes": args.archive.stat().st_size,
        "archive_sha256": sha256(args.archive),
        "split_rule": "sorted scenes; random.Random(0).sample(scene_names, 1090) for val",
        "selection": {
            "split": args.split,
            "count": args.count,
            "seed": args.seed,
            "method": "proportional stratified sample by top-level sensor family",
            "pool_sensor_counts": dict(sorted(Counter(PurePosixPath(scene).relative_to('SUNRGBD').parts[0] for scene in pool).items())),
            "selected_sensor_counts": dict(sorted(Counter(record['sensor'] for record in records).items())),
            "allocations": dict(sorted(allocations.items())),
        },
        "all_image_depth_shapes_match": all(record["shape_match"] for record in records),
        "records": records,
    }
    (args.output / "selected.txt").write_text("\n".join(record["image"] for record in records) + "\n", encoding="utf-8")
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "split": args.split,
        "selected": len(records),
        "sensor_counts": manifest["selection"]["selected_sensor_counts"],
        "shape_matches": sum(record["shape_match"] for record in records),
        "output": str(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
