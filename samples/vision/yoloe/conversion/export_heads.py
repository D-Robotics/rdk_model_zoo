# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Dense PF export adapters; preserve the checkpoint and upstream backbone routing."""

import torch
from torch import nn
import torch.nn.functional as F

VARIANTS = ("11s", "11m", "11l", "26n", "26s", "26m", "26l", "26x")
OUTPUT_NAMES = tuple(
    f"{kind}_{stride}" for stride in (8, 16, 32) for kind in ("cls", "box", "mces")
) + ("protos",)


def validate_head(head, variant):
    """Validate structural PF semantics before taking any raw export branch."""
    if variant not in VARIANTS:
        raise ValueError(f"Unsupported PF variant: {variant}")
    direct = variant.startswith("26")
    expected = {
        "nl": 3,
        "nm": 32,
        "nc": 4585,
        "reg_max": 1 if direct else 16,
        "end2end": direct,
    }
    if any(getattr(head, name, None) != value for name, value in expected.items()):
        raise ValueError(f"Expected fused {variant} PF head: {expected}")
    if tuple(getattr(head, "stride", torch.empty(0)).tolist()) != (8, 16, 32):
        raise ValueError("PF head requires strides 8, 16, 32.")
    branches = getattr(head, "lrpc", ())
    if len(branches) != 3:
        raise ValueError(
            "Expected three fused PF LRPC branches; prompt models are unsupported."
        )
    for branch in branches:
        vocab = getattr(branch, "vocab", None)
        if isinstance(vocab, nn.Linear):
            valid = vocab.out_features == 4585
        elif isinstance(vocab, nn.Conv2d):
            valid = (
                vocab.out_channels == 4585
                and vocab.kernel_size == (1, 1)
                and vocab.groups == 1
            )
        else:
            valid = False
        if not valid or not isinstance(getattr(branch, "loc", None), nn.Module):
            raise ValueError(
                "Expected 4585-class Linear or dense 1x1 vocabulary and localization module."
            )
    prefix = "one2one_" if direct else ""
    for suffix in ("cv2", "cv3", "cv5"):
        if len(getattr(head, prefix + suffix, ())) != 3:
            raise ValueError(f"Missing three-scale branch: {prefix+suffix}")
    if not isinstance(getattr(head, "proto", None), nn.Module):
        raise ValueError("Missing prototype branch.")


class RawPFHead(nn.Module):
    """Return NHWC logits, box channels, coefficients and prototypes, without filtering."""

    def __init__(self, head, variant):
        super().__init__()
        validate_head(head, variant)
        self.head = head
        self.direct = variant.startswith("26")

    def forward(self, features):
        head = self.head
        prefix = "one2one_" if self.direct else ""
        outputs = []
        for i in range(3):
            feature = features[i]
            cls = getattr(head, prefix + "cv3")[i](feature)
            loc = getattr(head, prefix + "cv2")[i](feature)
            vocab = head.lrpc[i].vocab
            scores = (
                F.conv2d(cls, vocab.weight[:, :, None, None], vocab.bias)
                if isinstance(vocab, nn.Linear)
                else vocab(cls)
            )
            boxes = head.lrpc[i].loc(loc)
            coefficients = getattr(head, prefix + "cv5")[i](feature)
            outputs.extend(
                value.permute(0, 2, 3, 1).contiguous()
                for value in (scores, boxes, coefficients)
            )
        proto = (
            head.proto(features[:3], return_semantic=False)
            if self.direct
            else head.proto(features[0])
        )
        outputs.append(proto.permute(0, 2, 3, 1).contiguous())
        return tuple(outputs)


class RawPFModel(nn.Module):
    """Keep model.<index> node names and the source model's saved-layer graph routing."""

    def __init__(self, network, variant):
        super().__init__()
        self.model = network.model
        self.save = set(network.save)
        self.head_from = network.model[-1].f
        self.raw_head = RawPFHead(network.model[-1], variant)

    def features(self, x):
        saved = []
        for layer in self.model[:-1]:
            if layer.f != -1:
                x = (
                    saved[layer.f]
                    if isinstance(layer.f, int)
                    else [x if i == -1 else saved[i] for i in layer.f]
                )
            x = layer(x)
            saved.append(x if layer.i in self.save else None)
        return [x if i == -1 else saved[i] for i in self.head_from]

    def forward(self, x):
        return self.raw_head(self.features(x))


def compare_selected_rows(actual, reference, actual_keys, reference_keys):
    """Require the same selected identities; compare values by identity, record order drift.

    This never accepts an added/dropped anchor or class as a near tie. Raw export
    outputs do not contain Top-K order, so that order is reported separately.
    """
    actual_keys, reference_keys = actual_keys.flatten(), reference_keys.flatten()
    if (
        actual_keys.numel() != reference_keys.numel()
        or actual_keys.unique().numel() != actual_keys.numel()
        or reference_keys.unique().numel() != reference_keys.numel()
    ):
        raise ValueError("Top-K identities must be unique and equal in count.")
    actual_order, reference_order = actual_keys.argsort(), reference_keys.argsort()
    if not torch.equal(actual_keys[actual_order], reference_keys[reference_order]):
        raise ValueError(
            "Top-K selected anchor/class set differs; near-boundary substitutions are not accepted."
        )
    torch.testing.assert_close(
        actual[actual_order], reference[reference_order], rtol=1e-4, atol=1e-4
    )
    changed = int(torch.count_nonzero(actual_keys != reference_keys))
    return {
        "selected_anchor_class_set": "identical",
        "selected_rows": actual_keys.numel(),
        "order_identical": changed == 0,
        "reordered_rows": changed,
        "comparison": "per-identity rtol=atol=1e-4",
    }


def reference_outputs(wrapper, sample):
    """Check all raw/decoded anchors against upstream; inspect Top-K identities separately."""
    with torch.inference_mode():
        features = wrapper.features(sample)
        expected = wrapper.raw_head(features)
        head = wrapper.raw_head.head
        flags = {
            name: (name in head.__dict__, getattr(head, name, None))
            for name in ("export", "dynamic", "format", "agnostic_nms")
        }
        inference_owned = "_inference" in head.__dict__
        original_inference = head._inference
        captured = {}

        def capture(raw):
            captured["raw"] = raw
            captured["decoded"] = original_inference(raw)
            return captured["decoded"]

        try:
            head.export, head.dynamic, head.format, head.agnostic_nms = (
                True,
                False,
                "onnx",
                True,
            )
            head._inference = capture
            upstream = head(list(features))
            raw = {
                "boxes": torch.cat(
                    [
                        expected[i].permute(0, 3, 1, 2).reshape(1, head.reg_max * 4, -1)
                        for i in (1, 4, 7)
                    ],
                    dim=2,
                ),
                "scores": torch.cat(
                    [
                        expected[i].permute(0, 3, 1, 2).reshape(1, 4585, -1)
                        for i in (0, 3, 6)
                    ],
                    dim=2,
                ),
                "mask_coefficient": torch.cat(
                    [
                        expected[i].permute(0, 3, 1, 2).reshape(1, 32, -1)
                        for i in (2, 5, 8)
                    ],
                    dim=2,
                ),
                "feats": features,
                "index": None,
            }
            errors = {}
            for name in ("boxes", "scores", "mask_coefficient"):
                torch.testing.assert_close(
                    raw[name], captured["raw"][name], rtol=1e-4, atol=1e-4
                )
                errors[name] = float((raw[name] - captured["raw"][name]).abs().max())
            decoded = original_inference(raw)
            torch.testing.assert_close(
                decoded, captured["decoded"], rtol=1e-4, atol=1e-4
            )
            torch.testing.assert_close(
                expected[-1].permute(0, 3, 1, 2), upstream[1], rtol=1e-4, atol=1e-4
            )
            comparison = {
                "all_raw_anchors": "passed",
                "decoded_before_topk": "passed",
                "raw_max_abs_error": errors,
                "topk": "not-applicable",
            }
            if head.end2end:
                _, classes_a, indices_a = head.get_topk_index(
                    decoded[:, 4:4589, :].permute(0, 2, 1), head.max_det
                )
                _, classes_b, indices_b = head.get_topk_index(
                    captured["decoded"][:, 4:4589, :].permute(0, 2, 1), head.max_det
                )
                rows = head.postprocess(decoded.permute(0, 2, 1))
                comparison["topk"] = compare_selected_rows(
                    rows[0],
                    upstream[0][0],
                    indices_a.long() * 4585 + classes_a.long(),
                    indices_b.long() * 4585 + classes_b.long(),
                )
            else:
                torch.testing.assert_close(decoded, upstream[0], rtol=1e-4, atol=1e-4)
        finally:
            if inference_owned:
                head._inference = original_inference
            elif "_inference" in head.__dict__:
                delattr(head, "_inference")
            for name, (owned, value) in flags.items():
                if owned:
                    setattr(head, name, value)
                elif name in head.__dict__:
                    delattr(head, name)
    return expected, comparison
