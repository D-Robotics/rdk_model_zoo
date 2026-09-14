#!/usr/bin/env python3
# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E402
"""Export one YOLOE-26 prompt-free checkpoint to the ten-output raw PF contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import ultralytics
from ultralytics import YOLOE
from ultralytics.nn.modules.head import LRPCHead, YOLOESegment26

SAMPLE = Path(__file__).resolve().parents[2]
RUNTIME = SAMPLE / "runtime" / "python"
sys.path.insert(0, str(RUNTIME))
from yoloe26seg import (  # noqa: E402
    CLASSES,
    OUTPUT_NAMES,
    OUTPUT_SHAPES,
    SIZES,
    STRIDES,
    model_stem,
    prepare_rgb,
    sha256,
    validate_shapes,
)


def validate_head(head: nn.Module) -> None:
    if not isinstance(head, YOLOESegment26) or not hasattr(head, "lrpc"):
        raise ValueError("Expected YOLOESegment26 with fused PF LRPC; text/visual heads are unsupported")
    if head.reg_max != 1 or not head.end2end or head.nl != 3 or head.nm != 32 or head.nc != CLASSES:
        raise ValueError("Expected end2end=True, reg_max=1, nl=3, nm=32, nc=4585")
    if tuple(head.stride.tolist()) != STRIDES:
        raise ValueError("Expected strides 8, 16, 32")
    for lrpc in head.lrpc:
        if not isinstance(lrpc, LRPCHead) or not isinstance(lrpc.vocab, (nn.Linear, nn.Conv2d)):
            raise ValueError("Unsupported LRPC vocabulary layer")
        if isinstance(lrpc.vocab, nn.Conv2d) and (lrpc.vocab.kernel_size != (1, 1) or lrpc.vocab.groups != 1):
            raise ValueError("Expected a dense 1x1 vocabulary convolution")


class RawPFHead(nn.Module):
    """Expose the static PF raw branches without upstream post-processing."""

    def __init__(self, head: nn.Module):
        super().__init__()
        validate_head(head)
        self.head = head

    def forward(self, features: list[torch.Tensor]) -> tuple[torch.Tensor, ...]:
        head = self.head
        outputs = []
        for index in range(3):
            feature = features[index]
            cls_feature = head.one2one_cv3[index](feature)
            loc_feature = head.one2one_cv2[index](feature)
            vocab = head.lrpc[index].vocab
            if isinstance(vocab, nn.Linear):
                scores = F.conv2d(cls_feature, vocab.weight[:, :, None, None], vocab.bias)
            else:
                scores = vocab(cls_feature)
            boxes = head.lrpc[index].loc(loc_feature)
            mask_coefficients = head.one2one_cv5[index](feature)
            outputs.extend(
                tensor.permute(0, 2, 3, 1)
                for tensor in (scores, boxes, mask_coefficients)
            )
        proto = head.proto(features[:3], return_semantic=False)
        outputs.append(proto.permute(0, 2, 3, 1))
        return tuple(outputs)


class RawPFModel(nn.Module):
    """Run the original backbone/neck and replace only the final PF head output."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.layers = model.model[:-1]
        self.save = model.save
        self.head_from = model.model[-1].f
        self.head = RawPFHead(model.model[-1])

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        saved = []
        for layer in self.layers:
            if layer.f != -1:
                x = saved[layer.f] if isinstance(layer.f, int) else [
                    x if index == -1 else saved[index] for index in layer.f
                ]
            x = layer(x)
            saved.append(x if layer.i in self.save else None)
        features = [x if index == -1 else saved[index] for index in self.head_from]
        return self.head(features)


def _reference_outputs(wrapper: RawPFModel, sample: torch.Tensor) -> tuple[torch.Tensor, ...]:
    with torch.inference_mode():
        expected = wrapper(sample)
        validate_shapes([tensor.shape for tensor in expected])
        captured = []
        hook = wrapper.head.register_forward_pre_hook(lambda _module, inputs: captured.append(inputs[0]))
        wrapper(sample)
        hook.remove()
        head = wrapper.head.head
        head.export, head.dynamic, head.format = True, False, "onnx"
        head.agnostic_nms = True
        upstream = head(captured[0])
        raw = {
            "boxes": torch.cat([
                expected[index].permute(0, 3, 1, 2).reshape(1, 4, -1)
                for index in (1, 4, 7)
            ], dim=2),
            "scores": torch.cat([
                expected[index].permute(0, 3, 1, 2).reshape(1, CLASSES, -1)
                for index in (0, 3, 6)
            ], dim=2),
            "mask_coefficient": torch.cat([
                expected[index].permute(0, 3, 1, 2).reshape(1, 32, -1)
                for index in (2, 5, 8)
            ], dim=2),
            "feats": captured[0],
            "index": None,
        }
        decoded = head.postprocess(head._inference(raw).permute(0, 2, 1))
        torch.testing.assert_close(decoded, upstream[0], rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(expected[-1].permute(0, 3, 1, 2), upstream[1], rtol=1e-4, atol=1e-4)
    return expected


def export(weights: Path, size: str, output_dir: Path, test_image: Path | None = None) -> dict:
    import onnx
    import onnxruntime as ort

    if size not in SIZES:
        raise ValueError(f"Expected one of {SIZES}, got {size!r}")
    if not weights.is_file():
        raise FileNotFoundError(weights)
    if output_dir.exists():
        raise FileExistsError(f"Refusing to reuse export directory: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir()

    model = YOLOE(str(weights)).model.cpu().float().eval()
    if model.yaml.get("scale") != size:
        raise ValueError(f"Checkpoint scale {model.yaml.get('scale')!r} differs from --size={size}")
    head = model.model[-1]
    validate_head(head)
    names = model.names
    names = [names[index] for index in range(CLASSES)] if isinstance(names, dict) else list(names)
    if len(names) != CLASSES:
        raise ValueError("Invalid checkpoint vocabulary")

    wrapper = RawPFModel(model).eval()
    torch.manual_seed(0)
    if test_image is None:
        sample = torch.rand(1, 3, 640, 640)
        validation_input = "seeded_random"
    else:
        import cv2

        image = cv2.imread(str(test_image), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Unreadable test image: {test_image}")
        sample = torch.from_numpy(prepare_rgb(image))
        validation_input = str(test_image.resolve())

    onnx_path = output_dir / f"{model_stem(size)}.onnx"
    metadata_path = output_dir / f"{model_stem(size)}.json"
    with torch.inference_mode():
        expected = _reference_outputs(wrapper, sample)
        torch.onnx.export(
            wrapper,
            (sample,),
            str(onnx_path),
            input_names=["images"],
            output_names=list(OUTPUT_NAMES),
            opset_version=17,
            dynamo=False,
            do_constant_folding=True,
        )

    graph = onnx.load(str(onnx_path), load_external_data=False)
    onnx.checker.check_model(graph)
    if [value.name for value in graph.graph.output] != list(OUTPUT_NAMES):
        raise ValueError("Exported ONNX output names/order changed")
    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = 2
    session = ort.InferenceSession(str(onnx_path), session_options, providers=["CPUExecutionProvider"])
    actual = session.run(list(OUTPUT_NAMES), {"images": sample.numpy()})
    validate_shapes([value.shape for value in actual])
    errors = {}
    for name, reference, result in zip(OUTPUT_NAMES, expected, actual):
        reference = reference.numpy()
        np.testing.assert_allclose(result, reference, rtol=2e-3, atol=2e-3, err_msg=name)
        errors[name] = float(np.max(np.abs(result - reference)))

    metadata = {
        "protocol": "yoloe26-pf-raw-v1",
        "size": size,
        "march": "nash-e",
        "input_shape": [1, 3, 640, 640],
        "reg_max": 1,
        "end2end": True,
        "pf_conf": 0,
        "output_names": list(OUTPUT_NAMES),
        "output_shapes": [list(shape) for shape in OUTPUT_SHAPES],
        "output_dtype": "float32",
        "names": names,
        "ultralytics_version": ultralytics.__version__,
        "checkpoint_sha256": sha256(weights),
        "onnx_sha256": sha256(onnx_path),
        "validation": {
            "input": validation_input,
            "upstream_static_pf": "passed",
            "onnx_max_abs_error": errors,
            "hbm": "not_run",
            "board": "not_run",
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_dir / f"{model_stem(size)}.names").write_text("\n".join(names) + "\n", encoding="utf-8")
    print(f"Export and float checks passed: {onnx_path}", flush=True)
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--size", required=True, choices=SIZES)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--test-image", type=Path)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    export(args.weights.resolve(), args.size, args.output_dir.resolve(), args.test_image)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
