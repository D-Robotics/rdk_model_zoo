#!/usr/bin/env python3
"""Download 50 diverse real COCO val2017 images for Gemma4 Vision calibration.

Usage:
    python download_coco_images.py [--output-dir ../../calibration_data/images]
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import zipfile
from collections import defaultdict
from pathlib import Path
from urllib.request import urlretrieve

DEFAULT_OUT = Path(__file__).resolve().parent.parent.parent / "calibration_data" / "images"
ANN_ZIP = Path("/tmp/coco_annotations_trainval2017.zip")
ANN_JSON = Path("/tmp/instances_val2017.json")
COCO_IMG_BASE = "http://images.cocodataset.org/val2017"
TARGET_COUNT = 50
SEED = 42


def download_annotations() -> None:
    if ANN_JSON.exists() and ANN_JSON.stat().st_size > 1_000_000:
        print(f"Using cached {ANN_JSON}")
        return
    url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
    if not ANN_ZIP.exists():
        print(f"Downloading COCO annotations (~241MB): {url}")
        urlretrieve(url, ANN_ZIP)
    print("Extracting instances_val2017.json ...")
    with zipfile.ZipFile(ANN_ZIP) as zf:
        with zf.open("annotations/instances_val2017.json") as src:
            ANN_JSON.write_bytes(src.read())
    print(f"Saved {ANN_JSON} ({ANN_JSON.stat().st_size / 1e6:.1f} MB)")


def pick_diverse_images(n: int) -> list[dict]:
    with open(ANN_JSON, encoding="utf-8") as f:
        coco = json.load(f)

    id_to_cats: dict[int, set[int]] = defaultdict(set)
    for ann in coco["annotations"]:
        id_to_cats[ann["image_id"]].add(ann["category_id"])

    cat_names = {c["id"]: c["name"] for c in coco["categories"]}
    images = {img["id"]: img for img in coco["images"]}

    by_cat: dict[int, list[dict]] = defaultdict(list)
    for img_id, img in images.items():
        cats = id_to_cats.get(img_id)
        if not cats:
            continue
        primary = sorted(cats)[0]
        by_cat[primary].append(img)

    rng = random.Random(SEED)
    selected: list[dict] = []
    used_ids: set[int] = set()

    cat_ids = sorted(by_cat.keys(), key=lambda c: cat_names.get(c, str(c)))
    rng.shuffle(cat_ids)
    while len(selected) < n and cat_ids:
        progressed = False
        for cid in cat_ids:
            pool = [x for x in by_cat[cid] if x["id"] not in used_ids]
            if not pool:
                continue
            img = rng.choice(pool)
            selected.append(img)
            used_ids.add(img["id"])
            progressed = True
            if len(selected) >= n:
                break
        if not progressed:
            break

    if len(selected) < n:
        rest = [img for img in images.values() if img["id"] in id_to_cats and img["id"] not in used_ids]
        rng.shuffle(rest)
        for img in rest:
            selected.append(img)
            used_ids.add(img["id"])
            if len(selected) >= n:
                break

    n_cat_sets = len({frozenset(id_to_cats[i["id"]]) for i in selected})
    print(f"Selected {len(selected)} images across {n_cat_sets} category sets")
    return selected[:n]


def download_images(images: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    root = out_dir.parent
    manifest = []
    ok = 0
    for i, img in enumerate(images):
        fname = img["file_name"]
        url = f"{COCO_IMG_BASE}/{fname}"
        out_name = f"coco_{i:02d}_{fname}"
        out_path = out_dir / out_name
        if out_path.exists() and out_path.stat().st_size > 1000:
            print(f"  skip exists {out_name}")
            ok += 1
            manifest.append({"local": out_name, "coco_file": fname, "url": url, "width": img["width"], "height": img["height"]})
            continue
        try:
            print(f"  [{i+1}/{len(images)}] {url}")
            urlretrieve(url, out_path)
            ok += 1
            manifest.append({"local": out_name, "coco_file": fname, "url": url, "width": img["width"], "height": img["height"]})
        except Exception as e:
            print(f"  FAIL {fname}: {e}", file=sys.stderr)

    manifest_path = root / "images_coco_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump({"count": ok, "target": len(images), "images": manifest}, f, indent=2, ensure_ascii=False)
    print(f"Downloaded {ok}/{len(images)} -> {out_dir}")
    print(f"Manifest: {manifest_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Download COCO calibration images")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT, help="Output image directory")
    args = parser.parse_args()

    download_annotations()
    images = pick_diverse_images(TARGET_COUNT)
    download_images(images, args.output_dir)
    n = len(list(args.output_dir.glob("coco_*.jpg")))
    print(f"\nDone. {n} COCO images in {args.output_dir}")
    return 0 if n >= 30 else 1


if __name__ == "__main__":
    raise SystemExit(main())
