"""
Export lite-avatar face generator (TorchScript) to ONNX for PTQ.

lite_avatar.py:
  self.encoder   = torch.jit.load(net_encode.pt)   # ref image -> [f0,f1,f2,f3] pyramid
  self.generator = torch.jit.load(net_decode.pt)   # (pyramid[4], mouth_param[1,32]) -> mouth image
Encoder/decoder are decoupled: app pre-computes 150 ref-frame pyramids (encoder
once per ref frame), then per output frame runs decoder(pyramid, param).

I/O (probed):
  net_encode: [1,3,384,384] -> list[f0(1,3,384,384), f1(1,16,192,192),
                                   f2(1,32,96,96),   f3(1,64,48,48)]
  net_decode: (list[4], [1,32]) -> [1,3,384,384]

Both exported featuremap (pre-normalized float) so FP32 ONNX and HBM see the
same tensor for cosine parity.

Usage:
    python conversion/scripts/export_facegen.py --data_dir data/preload --out conversion
"""
import os, sys, json, argparse
import numpy as np
import torch

N_FEAT = 4


class EncodeWrap(torch.nn.Module):
    """Wrap net_encode so it returns a TUPLE (not a list). TorchScript lists
    trace to a single ONNX Sequence output; a tuple traces to N named outputs,
    which is what hb_compile + cosine need. We SCRIPT (not trace) the wrapper so
    the inner ScriptModule's ops get inlined into the graph."""
    def __init__(self, enc):
        super().__init__()
        self.enc = enc

    def forward(self, image):
        outs = self.enc(image)
        return outs[0], outs[1], outs[2], outs[3]


class DecodeWrap(torch.nn.Module):
    """Wrap net_decode to take N individual feature inputs (+ param) and pack
    them into the list the TorchScript generator expects. Scripted, not traced."""
    def __init__(self, dec):
        super().__init__()
        self.dec = dec

    def forward(self, f0, f1, f2, f3, mouth_param):
        return self.dec([f0, f1, f2, f3], mouth_param)


def probe_and_export_encode(path, out_onnx):
    enc = torch.jit.load(path).eval()
    x = torch.zeros(1, 3, 384, 384)
    y = enc(x)
    assert isinstance(y, (list, tuple)) and len(y) == N_FEAT, \
        f"net_encode expected list of {N_FEAT}, got {type(y)} len {len(y) if hasattr(y,'__len__') else 'n/a'}"
    shapes = [tuple(t.shape) for t in y]
    print(f"  [encode] input [1,3,384,384] -> {N_FEAT} features {shapes}")
    wrap = torch.jit.script(EncodeWrap(enc))
    torch.onnx.export(
        wrap, (x,), out_onnx,
        input_names=["image"],
        output_names=[f"f{i}" for i in range(N_FEAT)],
        opset_version=13, dynamic_axes=None,
    )
    print(f"  exported -> {out_onnx}")
    return shapes


def probe_and_export_decode(path, out_onnx, feat_shapes, n_param=32):
    dec = torch.jit.load(path).eval()
    feats = [torch.zeros(*s) for s in feat_shapes]
    param = torch.zeros(1, n_param)
    g = dec(feats, param)
    assert isinstance(g, torch.Tensor), f"net_decode expected tensor, got {type(g)}"
    print(f"  [decode] {N_FEAT} feats + param[1,{n_param}] -> output {tuple(g.shape)}")
    wrap = torch.jit.script(DecodeWrap(dec))
    torch.onnx.export(
        wrap, (*feats, param), out_onnx,
        input_names=[f"f{i}" for i in range(N_FEAT)] + ["mouth_param"],
        output_names=["mouth_image"],
        opset_version=13, dynamic_axes=None,
    )
    print(f"  exported -> {out_onnx}")
    return tuple(g.shape)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="data/preload")
    p.add_argument("--out", default="conversion")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)

    enc_path = os.path.join(a.data_dir, "net_encode.pt")
    dec_path = os.path.join(a.data_dir, "net_decode.pt")

    print("=== net_encode.pt ===")
    feat_shapes = probe_and_export_encode(enc_path, os.path.join(a.out, "net_encode.onnx"))

    print("\n=== net_decode.pt ===")
    img_shape = probe_and_export_decode(dec_path, os.path.join(a.out, "net_decode.onnx"), feat_shapes)

    manifest = {
        "encode_input_shape": [1, 3, 384, 384],
        "encode_output_shapes": [list(s) for s in feat_shapes],
        "decode_input_shapes": [list(s) for s in feat_shapes] + [[1, 32]],
        "decode_output_shape": list(img_shape),
        "n_feat": N_FEAT,
        "n_mouth_param": 32,
    }
    mp = os.path.join(a.out, "facegen_manifest.json")
    with open(mp, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nmanifest -> {mp}\n{manifest}")


if __name__ == "__main__":
    main()
