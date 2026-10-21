#!/usr/bin/env python3
"""Reject generated hero covers that are letterboxed or the wrong size.

Published detail heroes are 1672x941 and the bitmap fills the card. Pending
classification panels keep the same gray card matte as release/assets/resnet-cls.webp.
That matte is the published Top-K chrome, not a letterbox: the color has to
match the published edge and the thickness has to stay within a few pixels of
it. WebP compression shortens the measured run (fastvit's bottom edge is 35px
against the published 39px). A near-white strip, or any other uniform strip
thicker than 2px, fails. Detection, segmentation, and plain photos have no
matte exception.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

EXPECTED_SIZE = (1672, 941)
MAX_PLAIN_BORDER = 2
LINE_STD_MAX = 4.0
LINE_MEAN_DELTA = 4.0
MATTE_COLOR_DELTA = 3
MATTE_THICKNESS_DELTA = 4
NEAR_WHITE = 248
SIDES = ("top", "bottom", "left", "right")


def _lines(array, side):
    if side == "top":
        return array
    if side == "bottom":
        return array[::-1]
    transposed = np.transpose(array, (1, 0, 2))
    if side == "left":
        return transposed
    return transposed[::-1]


def edge_strips(array):
    """Width of the uniform run on each side, and the color of its outer line."""
    sides = {}
    for side in SIDES:
        lines = _lines(array, side)
        base = lines[0].astype(np.float32).mean(axis=0)
        count = 0
        for line in lines:
            sample = line.astype(np.float32)
            if float(sample.std()) > LINE_STD_MAX:
                break
            if float(np.abs(sample.mean(axis=0) - base).max()) > LINE_MEAN_DELTA:
                break
            count += 1
        sides[side] = {
            "thickness": count,
            "color": [int(round(float(channel))) for channel in base],
        }
    return sides


def _near_white(color):
    return min(color) >= NEAR_WHITE


def _same_published_matte(got, reference):
    if _near_white(got["color"]) or _near_white(reference["color"]):
        return False
    if abs(got["thickness"] - reference["thickness"]) > MATTE_THICKNESS_DELTA:
        return False
    return all(
        abs(left - right) <= MATTE_COLOR_DELTA
        for left, right in zip(got["color"], reference["color"])
    )


def problems_for(array, *, classification, reference_strips, expected_size=EXPECTED_SIZE):
    height, width = array.shape[:2]
    found = []
    if (width, height) != tuple(expected_size):
        found.append(f"size {width}x{height}, expected {expected_size[0]}x{expected_size[1]}")
    for side, strip in edge_strips(array).items():
        if strip["thickness"] <= MAX_PLAIN_BORDER:
            continue
        if classification and _same_published_matte(strip, reference_strips[side]):
            continue
        kind = "near-white" if _near_white(strip["color"]) else "uniform"
        found.append(f"{side} {kind} border {strip['thickness']}px color {strip['color']}")
    return found


def _load_rgb(path):
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def _classification(provenance_dir, stem):
    meta_path = Path(provenance_dir) / f"{stem}.json"
    if not meta_path.is_file():
        return False
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return isinstance(meta.get("topk"), list)


def check_covers(reference_path, heroes_dir, provenance_dir):
    reference = _load_rgb(reference_path)
    ref_height, ref_width = reference.shape[:2]
    if (ref_width, ref_height) != EXPECTED_SIZE:
        return [
            f"published reference is {ref_width}x{ref_height}, "
            f"expected {EXPECTED_SIZE[0]}x{EXPECTED_SIZE[1]}"
        ]
    reference_strips = edge_strips(reference)
    heroes = sorted(Path(heroes_dir).glob("*.webp"))
    if not heroes:
        return [f"no generated covers in {heroes_dir}"]
    problems = []
    for webp in heroes:
        array = _load_rgb(webp)
        classification = _classification(provenance_dir, webp.stem)
        for item in problems_for(array, classification=classification, reference_strips=reference_strips):
            problems.append(f"{webp.name}: {item}")
    return problems


def main(argv):
    if len(argv) != 3:
        print(
            "usage: check_cover_edges.py REFERENCE.webp HEROES_DIR PROVENANCE_DIR",
            file=sys.stderr,
        )
        return 2
    problems = check_covers(argv[0], argv[1], argv[2])
    if problems:
        print("\n".join(problems), file=sys.stderr)
        return 1
    print(f"PASS: {len(list(Path(argv[1]).glob('*.webp')))} generated covers are {EXPECTED_SIZE[0]}x{EXPECTED_SIZE[1]} without a letterbox")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
