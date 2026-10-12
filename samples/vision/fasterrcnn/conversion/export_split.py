#!/usr/bin/env python3
# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Export the two board graphs from the public TorchVision checkpoint.

Produces, next to this script (``--out``, default ``onnx/``):

* ``part_a_f0only.op16.onnx``  — S targets (opset 16)
* ``part_a_f0only.op11.onnx``  — X5 (opset 11; HardSwish decomposed)
* ``part_b_batch150.op11.onnx`` — X5 PartB: fc6 as Conv 7x7 over the
  [150,256,7,7] RoI batch (bayes-e caps Gemm-as-conv inputs at 8192
  channels; flattened 12544 exceeds it), fc7/predictors as Gemm
* ``part_b_packed.op16.onnx``  — S PartB: fc6/fc7/predictors as convs over
  the packed ROI bank [1,256,1050,7]

PartA returns only ``rpn_obj``, ``rpn_deltas`` and ``feat0``: the model's
MultiScaleRoIAlign reads feat0 for every ROI (level mapper k_min==k_max==5),
so feat1 is dead output.  Equivalence against the loaded TorchVision model
is asserted on random tensors before saving.

Run inside the export container (torch 2.6.0+cpu, torchvision 0.21.0+cpu,
onnx 1.15.0); see README.md for the exact environment.
"""
from __future__ import annotations

import argparse
import hashlib
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
import torch
import torch.nn as nn
import torch.nn.functional as F

CHECKPOINT_URL = ("https://download.pytorch.org/models/"
                  "fasterrcnn_mobilenet_v3_large_320_fpn-907ea3f9.pth")
CHECKPOINT_SHA256 = "907ea3f91ff92242bc1baea8049276a3e76bca48ce7560bd268cc029f37977b5"

INPUT_SIZE = 320
NUM_PROPOSALS = 150
NUM_CLASSES = 91


def checkpoint_path() -> Path:
    """Local checkpoint file, downloading it once into the torch-hub cache."""
    import urllib.request

    cached = Path(torch.hub.get_dir()) / "checkpoints" / CHECKPOINT_URL.rsplit("/", 1)[-1]
    if not cached.is_file():
        cached.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {CHECKPOINT_URL}")
        urllib.request.urlretrieve(CHECKPOINT_URL, cached)
    return cached


def load_model() -> torch.nn.Module:
    """Build the canonical pretrained model (``weights=COCO_V1``).

    ``weights=COCO_V1, weights_backbone=None`` is the public serving
    recipe: one checkpoint download, no redundant ImageNet backbone
    weights, and — because ``weights`` selects the *trained* architecture —
    the backbone norm layer is ``FrozenBatchNorm2d``, exactly as in the
    recorded evaluation graphs.  Building with both weights flags ``None``
    instead selects ``nn.BatchNorm2d`` and changes the exported graph
    structure; ``scripts-container/21_recipe_check.py`` verified the
    canonical recipe is bit-identical in parameters, norm classes and
    backbone/RPN/RoI-stage outputs to the recorded
    ``torch.load + load_state_dict`` construction.
    """
    from torchvision.models.detection import (
        FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
        fasterrcnn_mobilenet_v3_large_320_fpn,
    )

    model = fasterrcnn_mobilenet_v3_large_320_fpn(
        weights=FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.COCO_V1,
        weights_backbone=None)
    # Belt and braces: the weights enum must have loaded exactly the
    # verified public checkpoint bytes (same cache file, hash-gated in
    # main()).
    state = torch.load(checkpoint_path(), map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def verify_checkpoint_cache() -> str | None:
    """Return the sha256 of the cached checkpoint file, if present."""
    hub = Path(torch.hub.get_dir()) / "checkpoints"
    cached = hub / CHECKPOINT_URL.rsplit("/", 1)[-1]
    if not cached.is_file():
        return None
    h = hashlib.sha256()
    with open(cached, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class PartAF0Only(nn.Module):
    """backbone + FPN + RPN head; returns the RPN outputs and feat0 only."""

    def __init__(self, m):
        super().__init__()
        self.backbone = m.backbone
        self.rpn_head = m.rpn.head

    def forward(self, images):
        from torchvision.models.detection.rpn import concat_box_prediction_layers

        features = self.backbone(images)
        feature_list = list(features.values())
        objectness, pred_bbox_deltas = self.rpn_head(feature_list)
        obj_cat, deltas_cat = concat_box_prediction_layers(objectness, pred_bbox_deltas)
        return obj_cat, deltas_cat, features["0"]


class PartBBatch150(nn.Module):
    """TwoMLPHead + predictor with fc6 restructured as a 7x7 convolution."""

    def __init__(self, m):
        super().__init__()
        head = m.roi_heads.box_head
        pred = m.roi_heads.box_predictor
        self.fc6conv = nn.Conv2d(256, 1024, kernel_size=7, bias=True)
        with torch.no_grad():
            self.fc6conv.weight.copy_(head.fc6.weight.view(1024, 256, 7, 7))
            self.fc6conv.bias.copy_(head.fc6.bias)
        self.fc7 = head.fc7
        self.cls_score = pred.cls_score
        self.bbox_pred = pred.bbox_pred

    def forward(self, pooled):  # [150,256,7,7]
        x = F.relu(self.fc6conv(pooled)).flatten(1)  # [150,1024]
        x = F.relu(self.fc7(x))
        return self.cls_score(x), self.bbox_pred(x)


class PartBPacked(nn.Module):
    """Head + predictor as convolutions over the packed ROI bank."""

    def __init__(self, m):
        super().__init__()
        head = m.roi_heads.box_head
        pred = m.roi_heads.box_predictor
        self.fc6conv = nn.Conv2d(256, 1024, kernel_size=(7, 7), stride=(7, 1), bias=True)
        with torch.no_grad():
            self.fc6conv.weight.copy_(head.fc6.weight.view(1024, 256, 7, 7))
            self.fc6conv.bias.copy_(head.fc6.bias)
        self.fc7conv = nn.Conv2d(1024, 1024, kernel_size=1, bias=True)
        with torch.no_grad():
            self.fc7conv.weight.copy_(head.fc7.weight.view(1024, 1024, 1, 1))
            self.fc7conv.bias.copy_(head.fc7.bias)
        self.clsconv = nn.Conv2d(1024, NUM_CLASSES, kernel_size=1, bias=True)
        with torch.no_grad():
            self.clsconv.weight.copy_(pred.cls_score.weight.view(NUM_CLASSES, 1024, 1, 1))
            self.clsconv.bias.copy_(pred.cls_score.bias)
        self.regconv = nn.Conv2d(1024, NUM_CLASSES * 4, kernel_size=1, bias=True)
        with torch.no_grad():
            self.regconv.weight.copy_(pred.bbox_pred.weight.view(NUM_CLASSES * 4, 1024, 1, 1))
            self.regconv.bias.copy_(pred.bbox_pred.bias)

    def forward(self, bank):  # [1,256,1050,7]
        x = F.relu(self.fc6conv(bank))   # [1,1024,150,1]
        x = F.relu(self.fc7conv(x))
        return self.clsconv(x), self.regconv(x)


def pack_bank(pooled: torch.Tensor) -> torch.Tensor:
    """[150,256,7,7] -> [1,256,1050,7] (the S PartB input layout)."""
    return pooled.permute(1, 0, 2, 3).reshape(1, 256, NUM_PROPOSALS * 7, 7).contiguous()


def hardswish_to_op11(path_in: str, path_out: str) -> int:
    """Rewrite HardSwish (opset>=14) into x * clip(x+3, 0, 6) / 6 at opset 11."""
    m = onnx.load(path_in)
    g = m.graph
    rewrote = 0
    new_nodes = []
    for node in g.node:
        if node.op_type == "HardSwish" and not node.domain:
            xin, yout = node.input[0], node.output[0]
            base = f"hs{rewrote}"
            add3 = onnx.helper.make_node("Add", [xin, f"{base}_c3"], [f"{base}_a3"], name=f"{base}_add3")
            clip = onnx.helper.make_node(
                "Clip", [f"{base}_a3", f"{base}_c0", f"{base}_c6"], [f"{base}_cl"], name=f"{base}_clip")
            mul = onnx.helper.make_node("Mul", [xin, f"{base}_cl"], [f"{base}_m"], name=f"{base}_mul")
            div6 = onnx.helper.make_node("Div", [f"{base}_m", f"{base}_c6d"], [yout], name=f"{base}_div6")
            new_nodes.extend([add3, clip, mul, div6])
            g.initializer.extend([
                onnx.numpy_helper.from_array(np.array(3.0, dtype=np.float32), f"{base}_c3"),
                onnx.numpy_helper.from_array(np.array(0.0, dtype=np.float32), f"{base}_c0"),
                onnx.numpy_helper.from_array(np.array(6.0, dtype=np.float32), f"{base}_c6"),
                onnx.numpy_helper.from_array(np.array(6.0, dtype=np.float32), f"{base}_c6d"),
            ])
            rewrote += 1
        else:
            new_nodes.append(node)
    del g.node[:]
    g.node.extend(new_nodes)
    for imp in m.opset_import:
        if not imp.domain:
            imp.version = 11
    onnx.checker.check_model(m)
    onnx.save(m, path_out)
    return rewrote


def summarize(path: str) -> dict:
    m = onnx.load(path)
    ops = Counter(n.op_type for n in m.graph.node)
    return {
        "path": path,
        "opset": {imp.domain or "ai.onnx": imp.version for imp in m.opset_import},
        "nodes": len(m.graph.node),
        "op_histogram": dict(sorted(ops.items())),
        "inputs": [(i.name, [d.dim_value or d.dim_param for d in i.type.tensor_type.shape.dim])
                   for i in m.graph.input],
        "outputs": [(o.name, [d.dim_value or d.dim_param for d in o.type.tensor_type.shape.dim])
                    for o in m.graph.output],
        "file_size_bytes": Path(path).stat().st_size,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(Path(__file__).parent / "onnx"),
                        help="output directory for the ONNX graphs")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cached = verify_checkpoint_cache()
    model = load_model()
    cached = cached or verify_checkpoint_cache()
    if cached != CHECKPOINT_SHA256:
        raise SystemExit(
            f"checkpoint sha256 {cached} does not match the verified "
            f"{CHECKPOINT_SHA256}; remove the stale torch-hub cache entry")

    report: dict = {"checkpoint_sha256_cached": cached,
                    "torch": torch.__version__,
                    "torchvision": __import__("torchvision").__version__,
                    "parts": {}}

    # ---- export FIRST: running the model in torch before exporting flips
    # torch.onnx.export's Conv+BatchNorm folding on, which changes the graph
    # bytes.  Numeric equivalence is verified afterwards through onnxruntime
    # against freshly computed torch references.
    parta = PartAF0Only(model).eval()
    partb_batch = PartBBatch150(model).eval()
    partb_packed = PartBPacked(model).eval()

    dummy_norm = torch.zeros(1, 3, INPUT_SIZE, INPUT_SIZE)
    dummy_pooled = torch.zeros(NUM_PROPOSALS, 256, 7, 7)
    dummy_bank = torch.zeros(1, 256, NUM_PROPOSALS * 7, 7)

    pa16 = str(out / "part_a_f0only.op16.onnx")
    torch.onnx.export(parta, (dummy_norm,), pa16, opset_version=16,
                      do_constant_folding=True, input_names=["images"],
                      output_names=["rpn_obj", "rpn_deltas", "feat0"], dynamo=False)
    report["parts"]["part_a_f0only.op16"] = summarize(pa16)

    pa11 = str(out / "part_a_f0only.op11.onnx")
    try:
        torch.onnx.export(parta, (dummy_norm,), pa11, opset_version=11,
                          do_constant_folding=True, input_names=["images"],
                          output_names=["rpn_obj", "rpn_deltas", "feat0"], dynamo=False)
        report["parts"]["part_a_f0only.op11"] = summarize(pa11)
    except Exception:
        n = hardswish_to_op11(pa16, pa11)
        report["parts"]["part_a_f0only.op11"] = {**summarize(pa11),
                                                 "hardswish_decomposed_nodes": n}

    pb11 = str(out / "part_b_batch150.op11.onnx")
    torch.onnx.export(partb_batch, (dummy_pooled,), pb11, opset_version=11,
                      do_constant_folding=True, input_names=["pooled"],
                      output_names=["class_logits", "box_regression"], dynamo=False)
    report["parts"]["part_b_batch150.op11"] = summarize(pb11)

    pb16 = str(out / "part_b_packed.op16.onnx")
    torch.onnx.export(partb_packed, (dummy_bank,), pb16, opset_version=16,
                      do_constant_folding=True, input_names=["bank"],
                      output_names=["cls_packed", "reg_packed"], dynamo=False)
    report["parts"]["part_b_packed.op16"] = summarize(pb16)

    # ---- numeric equivalence of the saved graphs vs torch references ----
    import onnxruntime as ort

    torch.manual_seed(0)
    norm = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE)
    with torch.no_grad():
        from torchvision.models.detection.rpn import concat_box_prediction_layers
        ref_obj, ref_deltas = model.rpn.head(list(model.backbone(norm).values()))
        ref_obj_cat, ref_deltas_cat = concat_box_prediction_layers(ref_obj, ref_deltas)
        ref_feat0 = model.backbone(norm)["0"]
        pooled = torch.randn(NUM_PROPOSALS, 256, 7, 7)
        ref_cls, ref_reg = model.roi_heads.box_predictor(model.roi_heads.box_head(pooled))

    report["equivalence"] = {}
    for tag, path in (("part_a_f0only.op16", pa16), ("part_a_f0only.op11", pa11)):
        session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        obj, deltas, feat0 = (torch.from_numpy(v) for v in session.run(None, {"images": norm.numpy()}))
        report["equivalence"][tag] = {
            "obj_max_abs": float((obj - ref_obj_cat).abs().max()),
            "deltas_max_abs": float((deltas - ref_deltas_cat).abs().max()),
            "feat0_max_abs": float((feat0 - ref_feat0).abs().max()),
        }
    session = ort.InferenceSession(pb11, providers=["CPUExecutionProvider"])
    cls_b, reg_b = (torch.from_numpy(v) for v in session.run(None, {"pooled": pooled.numpy()}))
    report["equivalence"]["part_b_batch150.op11"] = {
        "class_logits_max_abs": float((cls_b - ref_cls).abs().max()),
        "box_regression_max_abs": float((reg_b - ref_reg).abs().max()),
    }
    session = ort.InferenceSession(pb16, providers=["CPUExecutionProvider"])
    cls_p, reg_p = (torch.from_numpy(v) for v in session.run(None, {"bank": pack_bank(pooled).numpy()}))
    report["equivalence"]["part_b_packed.op16"] = {
        "class_logits_max_abs": float((cls_p[0, :, :, 0].t() - ref_cls).abs().max()),
        "box_regression_max_abs": float((reg_p[0, :, :, 0].t() - ref_reg).abs().max()),
    }
    for tag, metrics in report["equivalence"].items():
        for key, value in metrics.items():
            if not value < 1e-3:
                raise SystemExit(f"equivalence failed: {tag} {key} = {value}")

    import json
    report_path = out / "export_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"wrote {len(report['parts'])} graphs to {out}; report: {report_path}")


if __name__ == "__main__":
    main()
