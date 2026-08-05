#!/usr/bin/env python3
"""Prepare and verify 50 deterministic real COCO val2017 calibration images."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import zipfile
from collections import defaultdict
from pathlib import Path
from urllib.request import urlretrieve

DEFAULT_OUT = Path(__file__).resolve().parent.parent.parent / "calibration_data" / "images"
ANN_ZIP = Path("/tmp/coco_annotations_trainval2017.zip")
ANN_JSON = Path("/tmp/instances_val2017.json")
COCO_IMG_BASE = "https://images.cocodataset.org/val2017"
TARGET_COUNT = 50
SEED = 42
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def sha256_file(path: Path) -> str:
    """Return a file's SHA256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_sha256(value: object) -> bool:
    """Return whether value is a lowercase or uppercase SHA256 digest."""
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdefABCDEF" for character in value)
    )


def looks_like_jpeg(path: Path) -> bool:
    """Check the JPEG signature used by all COCO val2017 image files."""
    try:
        with path.open("rb") as stream:
            return stream.read(2) == b"\xff\xd8"
    except OSError:
        return False


def load_manifest(manifest_path: Path) -> dict:
    """Load a JSON manifest, returning an empty mapping when unavailable."""
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return manifest if isinstance(manifest, dict) else {}


def valid_manifest_header(manifest: dict) -> bool:
    """Check the immutable identity of the deterministic calibration set."""
    entries = manifest.get("images")
    return (
        manifest.get("dataset") == "COCO"
        and manifest.get("split") == "val2017"
        and manifest.get("seed") == SEED
        and manifest.get("target") == TARGET_COUNT
        and manifest.get("count") == TARGET_COUNT
        and isinstance(entries, list)
        and len(entries) == TARGET_COUNT
    )


def verified_cached_entry(
    output_path: Path, previous_entry: object, expected_entry: dict
) -> bool:
    """Reuse a cached image only when its prior provenance and hash still match."""
    if not isinstance(previous_entry, dict):
        return False
    if any(previous_entry.get(key) != value for key, value in expected_entry.items()):
        return False
    expected_hash = previous_entry.get("sha256")
    return (
        output_path.is_file()
        and output_path.stat().st_size > 1000
        and looks_like_jpeg(output_path)
        and is_sha256(expected_hash)
        and sha256_file(output_path) == expected_hash
    )


def download_annotations() -> None:
    """Download and extract the COCO validation annotation file."""
    if ANN_JSON.exists() and ANN_JSON.stat().st_size > 1_000_000:
        print(f"Using cached {ANN_JSON}")
        return
    url = "https://images.cocodataset.org/annotations/annotations_trainval2017.zip"
    if not ANN_ZIP.exists():
        print(f"Downloading COCO annotations (~241 MB): {url}")
        urlretrieve(url, ANN_ZIP)
    print("Extracting instances_val2017.json ...")
    with zipfile.ZipFile(ANN_ZIP) as archive:
        with archive.open("annotations/instances_val2017.json") as source:
            ANN_JSON.write_bytes(source.read())
    print(f"Saved {ANN_JSON} ({ANN_JSON.stat().st_size / 1e6:.1f} MB)")


def pick_diverse_images(count: int) -> list[dict]:
    """Select a deterministic category-diverse subset of COCO images."""
    with ANN_JSON.open(encoding="utf-8") as stream:
        coco = json.load(stream)

    image_categories: dict[int, set[int]] = defaultdict(set)
    for annotation in coco["annotations"]:
        image_categories[annotation["image_id"]].add(annotation["category_id"])

    category_names = {category["id"]: category["name"] for category in coco["categories"]}
    images = {image["id"]: image for image in coco["images"]}
    images_by_category: dict[int, list[dict]] = defaultdict(list)
    for image_id, image in images.items():
        categories = image_categories.get(image_id)
        if categories:
            images_by_category[sorted(categories)[0]].append(image)

    rng = random.Random(SEED)
    selected: list[dict] = []
    used_ids: set[int] = set()
    category_ids = sorted(
        images_by_category,
        key=lambda category_id: category_names.get(category_id, str(category_id)),
    )
    rng.shuffle(category_ids)

    while len(selected) < count and category_ids:
        progressed = False
        for category_id in category_ids:
            pool = [
                image
                for image in images_by_category[category_id]
                if image["id"] not in used_ids
            ]
            if not pool:
                continue
            image = rng.choice(pool)
            selected.append(image)
            used_ids.add(image["id"])
            progressed = True
            if len(selected) >= count:
                break
        if not progressed:
            break

    if len(selected) < count:
        remaining = [
            image
            for image in images.values()
            if image["id"] in image_categories and image["id"] not in used_ids
        ]
        rng.shuffle(remaining)
        selected.extend(remaining[: count - len(selected)])

    category_sets = {
        frozenset(image_categories[image["id"]]) for image in selected[:count]
    }
    print(f"Selected {min(len(selected), count)} images across {len(category_sets)} category sets")
    return selected[:count]


def download_images(images: list[dict], output_dir: Path) -> None:
    """Download selected COCO images and write a provenance manifest."""
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir.parent / "images_coco_manifest.json"
    previous_manifest = load_manifest(manifest_path)
    previous_entries = {}
    if valid_manifest_header(previous_manifest):
        previous_entries = {
            entry.get("local"): entry
            for entry in previous_manifest["images"]
            if isinstance(entry, dict) and isinstance(entry.get("local"), str)
        }

    manifest: list[dict] = []
    for index, image in enumerate(images):
        coco_name = image["file_name"]
        url = f"{COCO_IMG_BASE}/{coco_name}"
        local_name = f"coco_{index:02d}_{coco_name}"
        output_path = output_dir / local_name
        expected_entry = {
            "local": local_name,
            "coco_file": coco_name,
            "url": url,
            "width": image["width"],
            "height": image["height"],
        }
        if verified_cached_entry(
            output_path, previous_entries.get(local_name), expected_entry
        ):
            print(f"  skip verified {local_name}")
        else:
            print(f"  [{index + 1}/{len(images)}] {url}")
            temporary_path = output_path.with_name(output_path.name + ".part")
            temporary_path.unlink(missing_ok=True)
            try:
                urlretrieve(url, temporary_path)
                if temporary_path.stat().st_size <= 1000 or not looks_like_jpeg(
                    temporary_path
                ):
                    raise RuntimeError(f"Downloaded file is not a valid COCO JPEG: {url}")
                temporary_path.replace(output_path)
            finally:
                temporary_path.unlink(missing_ok=True)
        manifest.append({**expected_entry, "sha256": sha256_file(output_path)})

    manifest_path.write_text(
        json.dumps(
            {
                "dataset": "COCO",
                "split": "val2017",
                "seed": SEED,
                "count": len(manifest),
                "target": TARGET_COUNT,
                "images": manifest,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Manifest: {manifest_path}")


def verify_images(output_dir: Path) -> bool:
    """Verify that the calibration directory contains only manifest-backed COCO images."""
    manifest_path = output_dir.parent / "images_coco_manifest.json"
    if not manifest_path.is_file():
        print(f"ERROR: missing COCO manifest: {manifest_path}", file=sys.stderr)
        return False
    manifest = load_manifest(manifest_path)
    if not manifest:
        print(f"ERROR: invalid COCO manifest: {manifest_path}", file=sys.stderr)
        return False
    if not valid_manifest_header(manifest):
        print(
            "ERROR: manifest must describe the deterministic 50-image "
            "COCO val2017 set (seed 42)",
            file=sys.stderr,
        )
        return False

    entries = manifest["images"]
    expected_names: set[str] = set()
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            print(f"ERROR: invalid COCO manifest entry: {entry}", file=sys.stderr)
            return False
        local_name = entry.get("local", "")
        coco_name = entry.get("coco_file", "")
        url = entry.get("url", "")
        width = entry.get("width")
        height = entry.get("height")
        expected_hash = entry.get("sha256")
        expected_local_name = f"coco_{index:02d}_{coco_name}"
        if (
            not isinstance(coco_name, str)
            or Path(coco_name).name != coco_name
            or not coco_name.endswith(".jpg")
            or local_name != expected_local_name
        ):
            print(f"ERROR: invalid COCO manifest entry: {entry}", file=sys.stderr)
            return False
        if url != f"{COCO_IMG_BASE}/{coco_name}":
            print(f"ERROR: non-COCO source URL: {url}", file=sys.stderr)
            return False
        if (
            not isinstance(width, int)
            or isinstance(width, bool)
            or width <= 0
            or not isinstance(height, int)
            or isinstance(height, bool)
            or height <= 0
        ):
            print(f"ERROR: invalid COCO image dimensions: {entry}", file=sys.stderr)
            return False
        if not is_sha256(expected_hash):
            print(f"ERROR: missing or invalid SHA256: {entry}", file=sys.stderr)
            return False
        image_path = output_dir / local_name
        if not image_path.is_file() or image_path.stat().st_size <= 1000:
            print(f"ERROR: missing or invalid calibration image: {image_path}", file=sys.stderr)
            return False
        if not looks_like_jpeg(image_path):
            print(f"ERROR: calibration image is not JPEG: {image_path}", file=sys.stderr)
            return False
        if sha256_file(image_path) != expected_hash:
            print(f"ERROR: SHA256 mismatch: {image_path}", file=sys.stderr)
            return False
        expected_names.add(local_name)

    actual_names = {
        path.name
        for path in output_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    }
    if actual_names != expected_names:
        extras = sorted(actual_names - expected_names)
        missing = sorted(expected_names - actual_names)
        print("ERROR: calibration image set differs from manifest", file=sys.stderr)
        if extras:
            print(f"  unexpected images: {extras}", file=sys.stderr)
        if missing:
            print(f"  missing images: {missing}", file=sys.stderr)
        return False

    print(
        f"Verified {len(expected_names)} real COCO val2017 calibration images: "
        f"{output_dir}"
    )
    return True


def main() -> int:
    """Download or verify the deterministic COCO calibration set."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Validate the existing manifest-backed image set without downloading",
    )
    args = parser.parse_args()

    if args.verify_only:
        return 0 if verify_images(args.output_dir) else 1

    download_annotations()
    download_images(pick_diverse_images(TARGET_COUNT), args.output_dir)
    return 0 if verify_images(args.output_dir) else 1


if __name__ == "__main__":
    raise SystemExit(main())
