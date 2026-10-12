#!/usr/bin/env python3
# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Build the PTQ calibration sets for the two split graphs.

For every calibration image (letterboxed to the 320x320 deployment canvas exactly like
the float reference: bilinear ``align_corners=False`` on the [0,1] RGB canvas,
zero padding bottom/right) this script emits:

* ``parta_x5/images_NNN.rgbchw`` — raw float32 [0,255] RGB NCHW (the X5
  toolchain applies mean/scale in the model input node)
* ``parta_s/images_NNN.npy``     — normalized float32 (the S toolchain input)
* ``partb/pooled_NNN.dat``       — model-induced RoI features, zero-padded to
  the fixed [150,256,7,7] PartB input, raw float32 (X5 makertbin reads
  headerless cal files)
* ``partb_packed_s/bank_NNN.npy``— the same features in the packed
  [1,256,1050,7] ROI-bank layout (S PartB graph calibration)

The RoI features are induced by the exported PartA graph itself (run through
onnxruntime) followed by the exact torchvision RPN selection and RoIAlign, so
the calibration content matches what the deployed pipeline feeds PartB.

Calibration sources: ``--images`` directories/files are always used; with
``--coco-urls`` (a URL-per-line file, e.g. ``recorded_calibration_urls.txt``)
the script downloads those JPEGs and keeps the ones where the float model
detects at least two persons.  ``--hflip`` (default) adds a horizontal flip of
every image, doubling the set.  The manifest records provenance for each
sample.

Run inside the export container (see README.md); requires torch,
torchvision, onnxruntime and PIL, plus ``export_split.py`` output.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from torchvision.models.detection.image_list import ImageList

from export_split import (
    CHECKPOINT_SHA256,
    INPUT_SIZE,
    NUM_PROPOSALS,
    load_model,
    pack_bank,
)

NUM_ANCHORS_PER_LEVEL = (1500, 1500, 375)
PERSON_LABEL = 1
MIN_PERSONS_PER_COCO_IMAGE = 2


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_image_tensor(path: Path) -> torch.Tensor:
    """JPEG/PNG -> float32 CHW in [0,1] (the torchvision ToTensor convention)."""
    from PIL import Image

    arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
    return torch.from_numpy(arr.copy()).permute(2, 0, 1).contiguous().float() / 255.0


def letterbox(chw: torch.Tensor, size: int = INPUT_SIZE) -> tuple[torch.Tensor, dict]:
    """Aspect-preserving bilinear resize into a zero-padded square canvas.

    Matches the float reference transform: ``align_corners=False`` bilinear
    on the [0,1] canvas, dimensions via ``round()`` (banker's), padding at
    the bottom/right.
    """
    _, h, w = chw.shape
    scale = min(size / w, size / h)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    resized = torch.nn.functional.interpolate(
        chw[None], size=(nh, nw), mode="bilinear", align_corners=False)[0]
    canvas = torch.zeros((3, size, size), dtype=chw.dtype)
    canvas[:, :nh, :nw] = resized
    return canvas, {"scale": float(scale), "orig_h": int(h), "orig_w": int(w),
                    "net_h": nh, "net_w": nw, "canvas": size}


def rpn_middle(model, obj_cat, deltas_cat, anchors, image_shapes):
    """Exact torchvision RPN selection (decode + per-level topk + NMS)."""
    with torch.no_grad():
        decoded = model.rpn.box_coder.decode(deltas_cat.detach(), [anchors])
        decoded = decoded.view(1, -1, 4)
        boxes, _ = model.rpn.filter_proposals(
            decoded, obj_cat, image_shapes, list(NUM_ANCHORS_PER_LEVEL))
    return boxes[0]


def pooled_pad(pooled: torch.Tensor) -> np.ndarray:
    """RoI features zero-padded to the fixed [150,256,7,7] PartB input."""
    n = pooled.shape[0]
    if n >= NUM_PROPOSALS:
        return pooled[:NUM_PROPOSALS].contiguous().numpy()
    out = torch.zeros((NUM_PROPOSALS,) + tuple(pooled.shape[1:]), dtype=pooled.dtype)
    out[:n] = pooled
    return out.numpy()


def collect_images(images_args: list[str], out_root: Path) -> list[tuple[Path, dict]]:
    """Resolve --images entries to (path, provenance) pairs."""
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    entries: list[tuple[Path, dict]] = []
    for arg in images_args:
        p = Path(arg)
        if p.is_dir():
            for f in sorted(p.iterdir()):
                if f.suffix.lower() in exts:
                    entries.append((f, {"source": f"user:{f.name}", "sha256": sha256_file(f)}))
        elif p.is_file():
            entries.append((p, {"source": f"user:{p.name}", "sha256": sha256_file(p)}))
        else:
            raise SystemExit(f"calibration image path not found: {p}")
    return entries


def collect_coco(urls_file: Path, raw_dir: Path, model) -> list[tuple[Path, dict]]:
    """Download the URL list, keeping images with >= 2 detected persons."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    kept: list[tuple[Path, dict]] = []
    for line in urls_file.read_text().splitlines():
        fields = line.split()
        if not fields or fields[0].startswith("#"):
            continue
        url, expected = fields[0], (fields[1] if len(fields) > 1 else None)
        cocoid = Path(url).stem
        dest = raw_dir / f"{cocoid}.jpg"
        if not dest.is_file():
            print(f"downloading {url}")
            urllib.request.urlretrieve(url, dest)
        digest = sha256_file(dest)
        if expected and digest != expected:
            dest.unlink()
            raise SystemExit(f"{url}: sha256 {digest} does not match the recorded {expected}")
        canvas, _ = letterbox(load_image_tensor(dest))
        with torch.no_grad():
            dets = model([canvas])[0]
        n_person = int(sum(1 for label in dets["labels"] if int(label) == PERSON_LABEL))
        if n_person < MIN_PERSONS_PER_COCO_IMAGE:
            print(f"  skip {cocoid}: {n_person} person detections")
            dest.unlink(missing_ok=True)
            continue
        kept.append((dest, {"source": f"coco:{cocoid}", "sha256": digest,
                            "persons_detected": n_person}))
    return kept


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", nargs="*", default=[],
                        help="calibration image files/directories (repo test images count)")
    parser.add_argument("--coco-urls", default=None, type=Path,
                        help="URL-per-line COCO JPEG list (see recorded_calibration_urls.txt)")
    parser.add_argument("--hflip", choices=("repo", "all", "none"), default="repo",
                        help="horizontal-flip augmentation scope: 'repo' flips "
                             "only --images entries (reproduces the recorded "
                             "22-sample set), 'all' flips every image, 'none' "
                             "disables flipping")
    parser.add_argument("--onnx", default=str(Path(__file__).parent / "onnx"),
                        help="export_split.py output directory")
    parser.add_argument("--out", default=str(Path(__file__).parent / "calib"),
                        help="calibration output root")
    args = parser.parse_args()

    if not args.images and args.coco_urls is None:
        raise SystemExit("no calibration source: pass --images and/or --coco-urls")

    model = load_model()
    onnx_dir = Path(args.onnx)
    out_root = Path(args.out)
    parta_onnx = onnx_dir / "part_a_f0only.op16.onnx"
    if not parta_onnx.is_file():
        raise SystemExit(f"missing {parta_onnx}; run export_split.py first")

    session = ort.InferenceSession(str(parta_onnx), providers=["CPUExecutionProvider"])

    # Anchors are input-shape dependent only; generate them from the model.
    with torch.no_grad():
        feature_list = list(model.backbone(torch.zeros(1, 3, INPUT_SIZE, INPUT_SIZE)).values())
        anchors = model.rpn.anchor_generator(
            ImageList(torch.zeros(1, 3, INPUT_SIZE, INPUT_SIZE), [(INPUT_SIZE, INPUT_SIZE)]),
            feature_list)[0]
    assert anchors.shape == (sum(NUM_ANCHORS_PER_LEVEL), 4), anchors.shape

    raw_entries = collect_images(args.images, out_root)
    if args.coco_urls is not None:
        raw_entries += collect_coco(args.coco_urls, out_root / "raw", model)

    canvases: list[tuple[torch.Tensor, dict]] = []
    for path, prov in raw_entries:
        canvas, meta = letterbox(load_image_tensor(path))
        record = {**prov, "letterbox": meta, "augmentation": "none"}
        canvases.append((canvas, record))
        repo_image = prov["source"].startswith("user:")
        if args.hflip == "all" or (args.hflip == "repo" and repo_image):
            canvases.append((torch.flip(canvas, dims=[2]),
                             {**prov, "letterbox": meta, "augmentation": "hflip"}))
    if not canvases:
        raise SystemExit("no usable calibration images")

    for sub in ("parta_x5", "parta_s", "partb", "partb_packed_s"):
        (out_root / sub).mkdir(parents=True, exist_ok=True)

    mean = torch.tensor(model.transform.image_mean).view(3, 1, 1)
    std = torch.tensor(model.transform.image_std).view(3, 1, 1)
    image_shapes = [(INPUT_SIZE, INPUT_SIZE)]
    manifest = {"samples": []}

    for idx, (canvas, prov) in enumerate(canvases):
        norm = ((canvas - mean) / std)[None].numpy().astype(np.float32)
        (canvas * 255.0)[None].numpy().astype(np.float32).tofile(
            out_root / "parta_x5" / f"images_{idx:03d}.rgbchw")
        np.save(out_root / "parta_s" / f"images_{idx:03d}.npy", norm)

        obj_cat, deltas_cat, feat0 = session.run(None, {"images": norm})
        rois = rpn_middle(model, torch.from_numpy(obj_cat),
                          torch.from_numpy(deltas_cat), anchors, image_shapes)
        # RoIAlign on feat0 only: proven bit-exact with the stock two-map
        # MultiScaleRoIAlign at this input size (every ROI maps to level 0).
        pooled = model.roi_heads.box_roi_pool(
            {"0": torch.from_numpy(feat0)}, [rois], image_shapes)
        pooled_np = pooled_pad(pooled)
        # X5 makertbin reads raw float32 cal files (no .npy header);
        # hb_compile on S reads npy.  Same content, two containers.
        pooled_np.tofile(out_root / "partb" / f"pooled_{idx:03d}.dat")
        np.save(out_root / "partb_packed_s" / f"bank_{idx:03d}.npy",
                pack_bank(torch.from_numpy(pooled_np)))

        manifest["samples"].append({"idx": idx, **prov,
                                    "proposals_after_nms": int(rois.shape[0])})
        print(f"[{idx:03d}] {prov['source']} ({prov['augmentation']}): "
              f"rois={rois.shape[0]}")

    manifest["checkpoint_sha256"] = CHECKPOINT_SHA256
    manifest["torch"] = torch.__version__
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {len(manifest['samples'])} calibration samples under {out_root}")


if __name__ == "__main__":
    main()
