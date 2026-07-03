#!/usr/bin/env python3
"""
PP-LiteSeg-STDC1 board-side inference using hbm_runtime.
Run on RDK X5 (firmware >= 3.5.0).

Usage:
    python3 infer_board.py --model model.bin --image test.jpg --output ../test_data/result.jpg
"""
import argparse
import os
import sys

import cv2
import numpy as np


# Cityscapes 19-class palette (BGR for OpenCV)
PALETTE_BGR = np.array([
    [128,  64, 128],  # 0  road
    [232,  35, 244],  # 1  sidewalk
    [ 70,  70,  70],  # 2  building
    [156, 102, 102],  # 3  wall
    [153, 153, 190],  # 4  fence
    [153, 153, 153],  # 5  pole
    [ 30, 170, 250],  # 6  traffic light
    [  0, 220, 220],  # 7  traffic sign
    [ 35, 142, 107],  # 8  vegetation
    [152, 251, 152],  # 9  terrain
    [180, 130,  70],  # 10 sky
    [ 60,  20, 220],  # 11 person
    [  0,   0, 255],  # 12 rider
    [142,   0,   0],  # 13 car
    [100,  60,   0],  # 14 truck
    [ 70,   0,   0],  # 15 bus
    [100,  80,   0],  # 16 train
    [230,   0,   0],  # 17 motorcycle
    [ 32,  11, 119],  # 18 bicycle
], dtype=np.uint8)

CLASS_NAMES = [
    'road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
    'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky',
    'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle',
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model',  required=True)
    p.add_argument('--image',  required=True)
    p.add_argument('--output', default='result.jpg')
    p.add_argument('--alpha',  type=float, default=0.55)
    return p.parse_args()


def to_nv12(bgr, w, h):
    """BGR ndarray -> NV12 ndarray shaped (h*3//2, w) uint8."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_LINEAR)
    yuv_i420 = cv2.cvtColor(resized, cv2.COLOR_RGB2YUV_I420)
    # I420 layout: Y (h*w) + U (h//2 * w//2) + V (h//2 * w//2)
    # NV12 layout: Y (h*w) + UV interleaved (h//2 * w)
    y = yuv_i420[:h, :]                              # (h, w)
    u = yuv_i420[h: h + h // 4].reshape(h // 2, w // 2)
    v = yuv_i420[h + h // 4:].reshape(h // 2, w // 2)
    uv = np.stack([u, v], axis=-1).reshape(h // 2, w)  # interleave U V
    nv12 = np.vstack([y, uv])                          # (h*3//2, w)
    return np.ascontiguousarray(nv12, dtype=np.uint8)


def colorize(seg, palette):
    h, w = seg.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for cid in range(len(palette)):
        out[seg == cid] = palette[cid]
    return out


def draw_legend(canvas, cls_ids, palette, names):
    box, pad = 18, 6
    font, fs, th = cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1
    leg_h = (box + pad) * len(cls_ids) + pad
    leg_w = 155
    leg = np.full((leg_h, leg_w, 3), 30, dtype=np.uint8)
    for i, cid in enumerate(cls_ids):
        y0 = pad + i * (box + pad)
        c = [int(x) for x in palette[cid]]
        cv2.rectangle(leg, (pad, y0), (pad + box, y0 + box), c, -1)
        cv2.putText(leg, names[cid], (pad + box + 4, y0 + box - 3), font, fs, (220, 220, 220), th)
    h, w = canvas.shape[:2]
    canvas[4: 4 + leg_h, w - leg_w - 4: w - 4] = leg
    return canvas


def main():
    args = parse_args()

    try:
        from hbm_runtime import HB_HBMRuntime
    except ImportError:
        sys.exit('ERROR: hbm_runtime not found — run on RDK X5 with firmware >= 3.5.0')

    W, H = 1024, 512

    print(f'Loading image: {args.image}')
    bgr = cv2.imread(args.image)
    if bgr is None:
        sys.exit(f'ERROR: cannot read {args.image}')

    print('Converting to NV12...')
    nv12 = to_nv12(bgr, W, H)   # shape (768, 1024)

    print('Loading model...')
    runtime = HB_HBMRuntime(args.model)
    mname = runtime.model_names[0]
    iname = runtime.input_names[mname][0]

    print(f'  model: {mname}')
    print(f'  input : {iname}  shape={runtime.input_shapes[mname][iname]}  dtype={runtime.input_dtypes[mname][iname].name}')
    oname = runtime.output_names[mname][0]
    print(f'  output: {oname}  shape={runtime.output_shapes[mname][oname]}  dtype={runtime.output_dtypes[mname][oname].name}')

    print('Running BPU inference...')
    results = runtime.run({iname: nv12})

    raw = results[mname][oname]   # (1, 512, 1024, 1)  int32
    seg = raw.squeeze().astype(np.int32)   # (512, 1024)

    unique = sorted(np.unique(seg).tolist())
    valid  = [c for c in unique if 0 <= c < len(CLASS_NAMES)]
    print(f'Detected {len(valid)} classes: {[CLASS_NAMES[c] for c in valid]}')

    # ── visualize ───────────────────────────────────────────────────────────
    seg_color  = colorize(seg, PALETTE_BGR)
    orig_rsz   = cv2.resize(bgr, (W, H), interpolation=cv2.INTER_LINEAR)
    overlay    = cv2.addWeighted(orig_rsz, 1 - args.alpha, seg_color, args.alpha, 0)
    overlay    = draw_legend(overlay, valid, PALETTE_BGR, CLASS_NAMES)

    div   = np.full((H, 3, 3), 60, dtype=np.uint8)
    panel = np.hstack([orig_rsz, div, overlay, div, seg_color])

    hdr = np.full((36, panel.shape[1], 3), 35, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(hdr, 'Original',              (10,      25), font, 0.65, (200, 200, 200), 1)
    cv2.putText(hdr, f'Overlay  alpha={args.alpha:.2f}', (W + 13,   25), font, 0.65, (200, 200, 200), 1)
    cv2.putText(hdr, 'Segmentation',          (2*W + 16, 25), font, 0.65, (200, 200, 200), 1)
    result = np.vstack([hdr, panel])

    cv2.imwrite(args.output, result)
    print(f'Saved → {args.output}  ({result.shape[1]}×{result.shape[0]})')


if __name__ == '__main__':
    main()
