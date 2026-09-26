# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Deterministic calibration with explicit target/profile and digest manifest."""

import argparse
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
from samples._shared.assets import sha256_file
from samples.vision.yolo26_depth.runtime.python.geometry import make_context

TARGETS = ("x5", "s100", "s100p", "s600")
VARIANTS = ("n", "s", "m", "l", "x")


def select_profile(target, variant, experimental_lite=False):
    if target not in TARGETS or variant not in VARIANTS:
        raise ValueError("Unknown target or variant")
    if experimental_lite and target == "x5":
        raise ValueError("X5 has no lite conversion recipe")
    return (
        "lite"
        if target != "x5" and (experimental_lite or variant in ("l", "x"))
        else "nv12"
    )


def prepare_tensor(image, target, profile):
    if (
        target not in TARGETS
        or profile not in ("nv12", "lite")
        or (target == "x5" and profile != "nv12")
    ):
        raise ValueError("Unsupported calibration target/profile")
    if (
        not isinstance(image, np.ndarray)
        or image.ndim != 3
        or image.shape[-1] != 3
        or image.dtype != np.uint8
    ):
        raise ValueError("Expected BGR uint8 HWC image")
    ctx = make_context(*image.shape[:2], profile, "n")
    if profile == "lite":
        prepared = cv2.resize(image, (768, 768), interpolation=cv2.INTER_LINEAR)
    else:
        w, h = 768 - ctx.left - ctx.right, 768 - ctx.top - ctx.bottom
        resized = (
            image
            if image.shape[:2] == (h, w)
            else cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
        )
        prepared = cv2.copyMakeBorder(
            resized,
            ctx.top,
            ctx.bottom,
            ctx.left,
            ctx.right,
            cv2.BORDER_CONSTANT,
            value=(114, 114, 114),
        )
    rgb = np.ascontiguousarray(prepared[:, :, ::-1].transpose(2, 0, 1))
    tensor = rgb if target == "x5" else rgb[None].astype(np.float32) / 255.0
    geometry = {
        "original_hw": list(image.shape[:2]),
        "padding_tblr": [ctx.top, ctx.bottom, ctx.left, ctx.right],
        "resize": "letterbox" if profile == "nv12" else "scale-fill",
    }
    return tensor, geometry


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target", choices=TARGETS, required=True)
    p.add_argument("--variant", choices=VARIANTS, default="n")
    p.add_argument(
        "--experimental-lite",
        action="store_true",
        help="Retained S n/s/m experiments, not published profiles",
    )
    p.add_argument("--images", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--count", type=int, default=100)
    p.add_argument("--seed", type=int, default=20260725)
    p.add_argument("--size", type=int, choices=(768,), default=768)
    p.add_argument("--manifest", type=Path)
    p.add_argument("--report", type=Path)
    args = p.parse_args(argv)
    profile = select_profile(args.target, args.variant, args.experimental_lite)
    if args.count <= 0:
        raise ValueError("count must be positive")
    images = args.images.expanduser().resolve()
    output = args.output.expanduser().resolve()
    manifest = (args.manifest or output.with_suffix(".json")).expanduser().resolve()
    report = (args.report or output.with_suffix(".md")).expanduser().resolve()
    if manifest == report or any(
        path.is_relative_to(output) for path in (manifest, report)
    ):
        raise ValueError(
            "Manifest and report must be distinct files outside the tensor directory"
        )
    for path in (output, manifest, report):
        if path.exists():
            raise FileExistsError(path)
    candidates = sorted(
        p
        for p in images.rglob("*")
        if p.is_file()
        and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    )
    if len(candidates) < args.count:
        raise ValueError(f"Need {args.count} images, found {len(candidates)}")
    output.mkdir(parents=True, exist_ok=False)
    records = []
    for index, source in enumerate(
        random.Random(args.seed).sample(candidates, args.count)
    ):
        image = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Cannot decode selected calibration image: {source}")
        tensor, geometry = prepare_tensor(image, args.target, profile)
        name = f"{index:04d}.bin" if args.target == "x5" else f"{index:04d}.npy"
        path = output / name
        if args.target == "x5":
            tensor.tofile(path)
        else:
            np.save(path, tensor, allow_pickle=False)
        records.append(
            {
                "source": str(source.relative_to(images)),
                "source_sha256": sha256_file(source),
                "output": name,
                "output_sha256": sha256_file(path),
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "min": float(tensor.min()),
                "max": float(tensor.max()),
                **geometry,
            }
        )
    data = {
        "schema_version": "1.0",
        "target": args.target,
        "profile": profile,
        "variant": args.variant,
        "experimental": args.target != "x5"
        and profile == "lite"
        and args.variant in ("n", "s", "m"),
        "size": 768,
        "source_root": str(images),
        "tensor_directory": str(output),
        "selection": {"seed": args.seed, "count": args.count},
        "records": records,
        "input_contract": {
            "color": "RGB",
            "dtype": "uint8" if args.target == "x5" else "float32",
            "layout": "CHW" if args.target == "x5" else "NCHW",
            "normalization": (
                "none; Mapper applies /255" if args.target == "x5" else "/255"
            ),
            "note": "NV12 calibration is direct RGB with matching geometry; not byte-identical to runtime YUV conversion",
        },
    }
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(data, indent=2) + "\n")
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        f"# Calibration preparation\n\nTarget: {args.target}; profile: {profile}; count: {args.count}; seed: {args.seed}.\n\n"
        f"Manifest: {manifest}\n\nNo model export, compiler or board validation was run.\n"
    )
    print(
        json.dumps({"manifest": str(manifest), "profile": profile, "count": args.count})
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
