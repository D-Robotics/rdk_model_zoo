"""Numerical export-head checks use real PyTorch, with explicit synthetic PF heads."""

import copy
import unittest
import torch
from torch import nn
from samples.vision.yoloe.conversion.export_heads import RawPFHead, validate_head


class Proto(nn.Module):
    def forward(self, features, return_semantic=False):
        x = features[0] if isinstance(features, list) else features
        return x[:, :1].expand(-1, 32, -1, -1)


class Head(nn.Module):
    def __init__(self, variant):
        super().__init__()
        self.nl, self.nm, self.nc = 3, 32, 4585
        self.reg_max = 1 if variant.startswith("26") else 16
        self.end2end = self.reg_max == 1
        self.stride = torch.tensor([8, 16, 32])
        prefix = "one2one_" if self.end2end else ""
        for name, channels in [("cv2", self.reg_max * 4), ("cv3", 3), ("cv5", 32)]:
            setattr(
                self,
                prefix + name,
                nn.ModuleList(nn.Conv2d(3, channels, 1) for _ in range(3)),
            )
        self.lrpc = nn.ModuleList()
        for i in range(3):
            branch = nn.Module()
            branch.vocab = nn.Linear(3, 4585) if i != 1 else nn.Conv2d(3, 4585, 1)
            branch.loc = nn.Identity()
            self.lrpc.append(branch)
        self.proto = Proto()


class ExportHeadTests(unittest.TestCase):
    def test_topk_identity_comparison_rejects_substitution_despite_close_scores(self):
        from samples.vision.yoloe.conversion.export_heads import compare_selected_rows

        rows = torch.tensor([[10.0, 0.5], [20.0, 0.5000001]])
        keys = torch.tensor([17, 28])
        result = compare_selected_rows(rows.flip(0), rows, keys.flip(0), keys)
        self.assertFalse(result["order_identical"])
        self.assertEqual(result["reordered_rows"], 2)
        with self.assertRaisesRegex(ValueError, "set differs"):
            compare_selected_rows(rows, rows, torch.tensor([17, 29]), keys)
        with self.assertRaises(AssertionError):
            compare_selected_rows(rows + 1, rows, keys, keys)
        with self.assertRaisesRegex(ValueError, "unique"):
            compare_selected_rows(rows, rows, torch.tensor([17, 17]), keys)

    def test_dense_vocabulary_matches_original_linear_without_mutation(self):
        torch.manual_seed(9)
        for variant in ("11s", "26n"):
            head = Head(variant).eval()
            original = copy.deepcopy(head.state_dict())
            features = [torch.randn(1, 3, n, n) for n in (4, 2, 1)]
            raw = RawPFHead(head, variant)(features)
            prefix = "one2one_" if variant.startswith("26") else ""
            for i, feature in enumerate(features):
                x = getattr(head, prefix + "cv3")[i](feature)
                vocab = head.lrpc[i].vocab
                expected = (
                    vocab(x.permute(0, 2, 3, 1))
                    if isinstance(vocab, nn.Linear)
                    else vocab(x).permute(0, 2, 3, 1)
                )
                torch.testing.assert_close(raw[3 * i], expected)
                torch.testing.assert_close(
                    raw[3 * i + 1],
                    getattr(head, prefix + "cv2")[i](feature).permute(0, 2, 3, 1),
                )
                torch.testing.assert_close(
                    raw[3 * i + 2],
                    getattr(head, prefix + "cv5")[i](feature).permute(0, 2, 3, 1),
                )
            self.assertEqual(len(raw), 10)
            self.assertIsInstance(head.lrpc[0].vocab, nn.Linear)
            for key, value in original.items():
                torch.testing.assert_close(
                    head.state_dict()[key], value, rtol=0, atol=0
                )

    def test_incompatible_heads_fail_before_export(self):
        for attribute, value in [
            ("nc", 80),
            ("reg_max", 16),
            ("end2end", False),
            ("stride", torch.tensor([4, 8, 16])),
        ]:
            head = Head("26n")
            setattr(head, attribute, value)
            with self.assertRaises(ValueError):
                validate_head(head, "26n")
        head = Head("11s")
        head.lrpc[1].vocab = nn.Conv2d(3, 4585, 3)
        with self.assertRaises(ValueError):
            validate_head(head, "11s")
        with self.assertRaises(ValueError):
            validate_head(Head("11s"), "11n")
