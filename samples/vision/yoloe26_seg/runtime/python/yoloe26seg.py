# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E402
"""Static PF inference. Raw outputs are NHWC FP32, with no DFL or NMS."""

from dataclasses import dataclass
from pathlib import Path
import sys
import hashlib
import json
import re

import cv2
import numpy as np

SAMPLE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SAMPLE.parents[2]))
SIZES = tuple("nsmlx")
STRIDES = (8, 16, 32)
CLASSES = 4585
OUTPUT_NAMES = tuple(f"{kind}_{stride}" for stride in STRIDES for kind in ("cls", "box", "mc")) + ("proto",)
OUTPUT_SHAPES = tuple((1, 640 // stride, 640 // stride, channels)
                      for stride in STRIDES for channels in (CLASSES, 4, 32)) + ((1, 160, 160, 32),)
MARCHES = ("nash-e", "nash-m")


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_stem(size):
    if size not in SIZES:
        raise ValueError(f"Unsupported model size: {size}")
    return f"yoloe_26{size}_seg_pf"


def hbm_name(size, march="nash-e"):
    if march not in MARCHES:
        raise ValueError("Only S100/nash-e and S100P/nash-m are supported")
    return f"{model_stem(size)}_{march.replace('-', '')}_640x640_nv12.hbm"


def detect_march():
    path = Path("/sys/class/boardinfo/soc_name")
    soc = path.read_text().strip().lower() if path.exists() else ""
    board_path = path.with_name("board_type")
    board = board_path.read_text().strip().lower() if board_path.exists() else ""
    board = re.sub(r"[^a-z0-9]", "", board)
    if soc not in ("s100", "s100p"):
        raise RuntimeError(f"Only S100/S100P are supported; detected {soc or 'unknown'}")
    if soc == "s100p" or board in ("p", "nashm") or board.startswith(("s100p", "rdks100p")):
        return "nash-m"
    return "nash-e"


def validate_shapes(shapes):
    if tuple(tuple(int(n) for n in shape) for shape in shapes) != OUTPUT_SHAPES:
        raise ValueError("Expected the ten 640x640 YOLOE-26 PF NHWC outputs")


def read_metadata(path):
    with open(path, encoding="utf-8") as stream:
        data = json.load(stream)
    model_stem(data["size"])
    if data.get("protocol") != "yoloe26-pf-raw-v1" or data.get("march") not in MARCHES:
        raise ValueError("Expected S100/S100P YOLOE-26 PF metadata")
    if data.get("input_shape") != [1, 3, 640, 640] or data.get("reg_max") != 1:
        raise ValueError("Only batch=1, 640x640, reg_max=1 is supported")
    if data.get("end2end") is not True or data.get("pf_conf") != 0:
        raise ValueError("Only static PF end-to-end models are supported")
    if data.get("output_names") != list(OUTPUT_NAMES) or data.get("vocab_chunk_size", 0) != 0:
        raise ValueError("Only the released ten-output contract is supported")
    validate_shapes(data["output_shapes"])
    if len(data.get("names", [])) != CLASSES:
        raise ValueError("Expected all 4585 checkpoint-ordered labels")
    return data


def letterbox(image):
    if image is None or image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("Expected a nonempty uint8 BGR image")
    h, w = image.shape[:2]
    if not h or not w:
        raise ValueError("Empty image")
    gain = min(640 / h, 640 / w)
    width, height = max(1, round(w * gain)), max(1, round(h * gain))
    left, top = (640 - width) // 2, (640 - height) // 2
    resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
    padded = cv2.copyMakeBorder(resized, top, 640 - height - top,
                               left, 640 - width - left, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return padded, (gain, left, top, width, height)


def prepare_rgb(image):
    padded, _ = letterbox(image)
    return np.ascontiguousarray(padded[..., ::-1].transpose(2, 0, 1)[None], dtype=np.float32) / 255


def topk_indices(values, count):
    """Stable descending ordering, with original index as the tie breaker."""
    values = np.asarray(values).reshape(-1)
    count = min(count, values.size)
    if count == 0:
        return np.empty(0, dtype=np.int64)
    threshold = np.partition(values, values.size - count)[values.size - count]
    higher = np.flatnonzero(values > threshold)
    tied = np.flatnonzero(values == threshold)[:count - higher.size]
    selected = np.concatenate((higher, tied))
    return selected[np.lexsort((selected, -values[selected]))]


def dequantize_output(value, quant):
    """Normalize HBM outputs to logical FP32, preserving scalar zero points."""
    if value.dtype == np.float32:
        return value
    if value.dtype not in (np.int8, np.uint8, np.int16, np.uint16, np.int32):
        raise ValueError(f"Unsupported output dtype: {value.dtype}")
    quant_type = getattr(quant.quant_type, "name", str(quant.quant_type))
    if quant_type not in ("SCALE", "1"):
        raise ValueError("Integer output has no supported SCALE quantization metadata")
    scale = np.asarray(quant.scale, dtype=np.float32).reshape(-1)
    zero = np.asarray(quant.zero_point, dtype=np.float32).reshape(-1)
    if not scale.size or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("Invalid quantization scales")
    if not zero.size:
        zero = np.zeros(1, dtype=np.float32)
    if not np.isfinite(zero).all():
        raise ValueError("Invalid zero points")
    if scale.size > 1 or zero.size > 1:
        axis = int(quant.axis)
        if not -value.ndim <= axis < value.ndim:
            raise ValueError("Invalid quantization axis")
        axis %= value.ndim
        shape = [1] * value.ndim
        shape[axis] = value.shape[axis]
        if scale.size not in (1, shape[axis]) or zero.size not in (1, shape[axis]):
            raise ValueError("Scale/zero-point length does not match quantization axis")
        if scale.size > 1:
            scale = scale.reshape(shape)
        if zero.size > 1:
            zero = zero.reshape(shape)
    return (value.astype(np.float32) - zero) * scale


def decode_candidates(outputs, score_threshold=0.25, max_det=300, single_label=True):
    """Match end-to-end top-k selection; never perform IoU suppression."""
    if not 0 < score_threshold < 1 or not 1 <= max_det <= 8400:
        raise ValueError("Require 0 < score threshold < 1 and 1 <= max_det <= 8400")
    validate_shapes([x.shape for x in outputs])
    if any(x.dtype != np.float32 for x in outputs):
        raise ValueError("Expected FP32 outputs after runtime dequantization")
    if any(not np.isfinite(x).all() for x in outputs):
        raise ValueError("Non-finite model output")
    # Select per-scale anchor candidates first to avoid copying the full vocabulary tensor.
    entries = []
    for scale, stride in enumerate(STRIDES):
        logits = outputs[3 * scale].reshape(-1, CLASSES)
        indices = topk_indices(logits.max(axis=1), max_det)
        for anchor in indices:
            entries.append((float(logits[anchor].max()), scale, int(anchor)))
    entries.sort(key=lambda item: (-item[0], item[1], item[2]))
    entries = entries[:max_det]
    logits = np.stack([outputs[3 * s].reshape(-1, CLASSES)[a] for _, s, a in entries])
    if single_label:
        labels = logits.argmax(axis=1)
        anchor_ids = np.arange(len(entries))
        values = logits[anchor_ids, labels]
    else:
        selected = topk_indices(logits, max_det)
        anchor_ids, labels = selected // CLASSES, selected % CLASSES
        values = logits.reshape(-1)[selected]
    keep = values > np.log(score_threshold / (1 - score_threshold))
    values, labels, anchor_ids = values[keep], labels[keep], anchor_ids[keep]
    boxes, coefficients = [], []
    for index in anchor_ids:
        _, scale, anchor = entries[index]
        stride = STRIDES[scale]
        grid = 640 // stride
        y, x = divmod(anchor, grid)
        left, top, right, bottom = outputs[3 * scale + 1][0, y, x]
        boxes.append([(x + .5 - left) * stride, (y + .5 - top) * stride,
                      (x + .5 + right) * stride, (y + .5 + bottom) * stride])
        coefficients.append(outputs[3 * scale + 2][0, y, x])
    scores = 1 / (1 + np.exp(-np.clip(values, -80, 80)))
    return (np.asarray(boxes, dtype=np.float32).reshape(-1, 4), scores.astype(np.float32),
            labels.astype(np.int64), np.asarray(coefficients, dtype=np.float32).reshape(-1, 32))


def restore_masks(boxes, coefficients, proto, original_shape):
    """Upsample then crop, matching upstream process_mask(upsample=True).

    Binary masks are unpadded and restored to the source image with nearest
    interpolation. No morphology or extra mask threshold is applied.
    """
    h, w = original_shape[:2]
    gain = min(640 / h, 640 / w)
    width, height = max(1, round(w * gain)), max(1, round(h * gain))
    left, top = (640 - width) // 2, (640 - height) // 2
    yy, xx = np.mgrid[:640, :640]
    masks = []
    for box, coeff in zip(boxes, coefficients):
        raw = proto @ coeff
        raw = cv2.resize(raw, (640, 640), interpolation=cv2.INTER_LINEAR)
        x1, y1, x2, y2 = box
        binary = ((raw > 0) & (xx >= x1) & (xx < x2) & (yy >= y1) & (yy < y2)).astype(np.uint8)
        binary = binary[top:top + height, left:left + width]
        masks.append(cv2.resize(binary, (w, h), interpolation=cv2.INTER_NEAREST).astype(bool))
    restored = boxes.copy()
    restored[:, [0, 2]] = np.clip((boxes[:, [0, 2]] - left) / gain, 0, w)
    restored[:, [1, 3]] = np.clip((boxes[:, [1, 3]] - top) / gain, 0, h)
    return restored, masks


@dataclass
class YoloE26SegConfig:
    model_path: str
    metadata_path: str
    score_thres: float = 0.25
    max_det: int = 300
    single_label: bool = True


class YoloE26Seg:
    def __init__(self, config):
        march = detect_march()
        import hbm_runtime

        self.cfg = config
        self.metadata = read_metadata(config.metadata_path)
        if self.metadata["march"] != march:
            raise ValueError(f"Model march {self.metadata['march']} does not match board {march}")
        if self.metadata.get("hbm_sha256") != sha256(config.model_path):
            raise ValueError("HBM hash missing or mismatched; use the matching downloaded metadata or conversion/mapper.py")
        self.model = hbm_runtime.HB_HBMRuntime(config.model_path)
        if len(self.model.model_names) != 1:
            raise ValueError("Expected a single-model HBM")
        self.model_name = self.model.model_names[0]
        self.input_names = self.model.input_names[self.model_name]
        shapes = self.model.input_shapes[self.model_name]
        expected = [(1, 640, 640, 1), (1, 320, 320, 2)]
        self.input_names = sorted(self.input_names, key=lambda name: -np.prod(shapes[name]))
        if [tuple(shapes[n]) for n in self.input_names] != expected:
            raise ValueError("Expected S100/S100P NV12 Y/UV inputs at 640x640")
        self.output_names = self.model.output_names[self.model_name]
        self.output_quants = self.model.output_quants[self.model_name]
        if list(self.output_names) != list(OUTPUT_NAMES):
            raise ValueError("HBM output names/order do not match export metadata")

    def forward(self, image):
        from utils.py_utils.preprocess import bgr_to_nv12_planes

        padded, _ = letterbox(image)
        y, uv = bgr_to_nv12_planes(padded)
        raw = self.model.run({self.model_name: dict(zip(self.input_names, (y, uv)))})[self.model_name]
        outputs = [dequantize_output(raw[name], self.output_quants[name]) for name in self.output_names]
        return outputs

    def post_process(self, outputs, original_shape):
        boxes, scores, labels, coefficients = decode_candidates(
            outputs, self.cfg.score_thres, self.cfg.max_det, self.cfg.single_label)
        boxes, masks = restore_masks(boxes, coefficients, outputs[-1][0], original_shape)
        return boxes, scores, labels, masks

    def predict(self, image):
        return self.post_process(self.forward(image), image.shape)

    __call__ = predict
