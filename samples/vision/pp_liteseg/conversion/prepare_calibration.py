#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare PP-LiteSeg calibration tensors for hb_mapper.")
    parser.add_argument("--src", type=Path, required=True, help="Directory containing calibration images.")
    parser.add_argument("--out", type=Path, default=Path("calibration_data_rgb_f32_1024x512"), help="Output directory.")
    parser.add_argument("--width", type=int, default=1024, help="Model input width.")
    parser.add_argument("--height", type=int, default=512, help="Model input height.")
    parser.add_argument("--num", type=int, default=50, help="Maximum number of images to export.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed used when sampling images.")
    return parser.parse_args()


def collect_images(src):
    suffixes = {".jpg", ".jpeg", ".png", ".bmp"}
    return sorted(path for path in src.rglob("*") if path.is_file() and path.suffix.lower() in suffixes)


def main():
    args = parse_args()
    if min(args.width, args.height, args.num) <= 0:
        raise ValueError("width, height and num must be positive")
    if not args.src.is_dir():
        raise NotADirectoryError(args.src)
    manifest_path = args.out.with_name(args.out.name + ".manifest.json")
    if args.out.exists() or manifest_path.exists():
        raise FileExistsError("Choose a new output directory; existing tensors or manifests are never overwritten")
    images = collect_images(args.src)
    if not images:
        raise FileNotFoundError(f"No calibration images found in {args.src}")

    if len(images) > args.num:
        rng = np.random.default_rng(args.seed)
        selected = rng.choice(len(images), args.num, replace=False)
        images = [images[index] for index in sorted(selected)]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".pp-cal-", dir=args.out.parent))
    samples = []
    try:
        for index, image_path in enumerate(images):
            source_bytes = image_path.read_bytes()
            image = cv2.imdecode(np.frombuffer(source_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"Unreadable calibration image: {image_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (args.width, args.height), interpolation=cv2.INTER_LINEAR)
            tensor = np.transpose(image, (2, 0, 1))[None].astype("<f4")
            relative = image_path.relative_to(args.src).as_posix()
            identity = hashlib.sha256(relative.encode()).hexdigest()[:12]
            name = f"{index:06d}_{identity}_{image_path.stem}.rgbchw"
            data = tensor.tobytes()
            (staging / name).write_bytes(data)
            samples.append({"source": relative, "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
                            "tensor": name, "tensor_sha256": hashlib.sha256(data).hexdigest(),
                            "bytes": len(data)})
        # The compiler directory contains tensors only; provenance is a sibling.
        manifest = {"source_directory": str(args.src.resolve()), "seed": args.seed,
                    "shape": [1, 3, args.height, args.width], "dtype": "<f4",
                    "normalization": "none; RGB values 0..255", "samples": samples}
        staging.rename(args.out)
        with manifest_path.open("x") as stream:
            json.dump(manifest, stream, indent=2)
            stream.write("\n")
    finally:
        if staging.exists():
            shutil.rmtree(staging)

    expected_bytes = 1 * 3 * args.height * args.width * 4
    print(f"Wrote {len(list(args.out.glob('*.rgbchw')))} calibration tensors to {args.out}")
    print(f"Expected file size: {expected_bytes} bytes per tensor")


if __name__ == "__main__":
    main()
