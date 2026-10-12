"""
Probe + export the lite-avatar paraformer ENCODER with all 50 hidden-state
outputs (the `intermediate_outs` that audio2mouth consumes as input_au).

lite-avatar's extract_paraformer_feature.py runs asr_model.encode(raw_audio)
which returns (enc, intermediate_outs) where intermediate_outs is a list of
50 per-block tensors [1, T, 512] (sanm encoder, num_blocks=50). audio2mouth's
input_au [1,30,50,512] = these 50 layers, linearly-interpolated to 30 frames.

This script:
  1. Builds the paraformer e2e model via funasr_local (like extract_paraformer_feature).
  2. Probes: runs e2e.encode on dummy audio, prints len(intermediate_outs) + shapes.
  3. Exports a wrapped encoder -> ONNX with a single stacked output
     `hidden_states` of shape [1, T, 50, 512] (input = post-CMVN featuremap
     [1, 400, 560], matching model-zoo encoder input_type_rt: featuremap).
  4. Verifies the ONNX with onnxruntime vs the PyTorch intermediate_outs.

Usage (from repo root, weights/ populated):
    python conversion/scripts/probe_paraformer_encoder.py
"""
import os, sys, argparse
import numpy as np
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
os.chdir(REPO_ROOT)
sys.path.insert(0, REPO_ROOT)


def build_s2t():
    from extract_paraformer_feature import Speech2Text, speech2text_kwargs
    s2t = Speech2Text(**speech2text_kwargs)
    return s2t


def probe(s2t, T_audio=16000 * 5):
    asr = s2t.asr_model
    asr.eval()
    speech = torch.tensor(np.zeros((1, T_audio), dtype=np.float32))
    speech_lengths = torch.tensor([T_audio])
    print(f"probe: audio {T_audio} samples")
    # e2e encode takes (speech, speech_lengths) — NO frame_cnt (that's a
    # Speech2Text.__call__ concept used only for post-encode interpolation).
    with torch.no_grad():
        enc_out, enc_len = asr.encode(speech, speech_lengths)
    if isinstance(enc_out, tuple):
        enc, hidden_states = enc_out
    else:
        enc, hidden_states = enc_out, None
    print(f"  enc (final) shape={tuple(enc.shape)}")
    if hidden_states is not None:
        print(f"  len(intermediate_outs)={len(hidden_states)}")
        for i, h in enumerate(hidden_states[:3]):
            shp = tuple(h.shape) if hasattr(h, "shape") else h
            typ = type(h).__name__
            print(f"    [{i}] {typ} shape={shp}")
        print(f"    ... last [{len(hidden_states)-1}] {type(hidden_states[-1]).__name__}")
    return asr, hidden_states, enc


class EncoderAvatar(torch.nn.Module):
    """Wrap sanm encoder to emit stacked intermediate_outs [B, T, L, D]."""
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, feats, feats_lengths=None):
        # feats_lengths optional: when None, derive from feats (concrete at trace
        # time -> the length-dependent mask folds to a constant, leaving a single
        # ONNX input `feats`).
        if feats_lengths is None:
            feats_lengths = torch.tensor([feats.shape[1]], device=feats.device)
        out = self.encoder(feats, feats_lengths)
        # sanm returns ((xs_pad, intermediate_outs), olens, None) when
        # intermediate_outs is non-empty
        if isinstance(out, tuple) and len(out) == 3 and isinstance(out[0], tuple):
            (xs_pad, intermediate_outs), olens, _ = out
        else:
            # fallback: out may be (xs_pad, olens, _)
            xs_pad, olens = out[0], out[1]
            intermediate_outs = [xs_pad]
        stacked = torch.stack(intermediate_outs, dim=1)   # [B, L, T, D]
        # transpose to [B, T, L, D] — matches audio2mouth input_au layout
        stacked = stacked.transpose(1, 2)
        return stacked


def export_encoder(asr, dst, T=400, feat_dim=560):
    # get a real post-CMVN featuremap, then PAD/truncate to T=400 so the
    # exported ONNX bakes the fixed 1x400x560 shape hb_compile expects
    # (mirrors model-zoo encoder_int16.yaml input_shape 1x400x560).
    speech = torch.tensor(np.zeros((1, T * 160), dtype=np.float32))
    speech_lengths = torch.tensor([speech.shape[1]])
    with torch.no_grad():
        feats, feats_lengths = asr._extract_feats(speech, speech_lengths)
    if feats.shape[1] < T:
        pad = torch.zeros(1, T - feats.shape[1], feat_dim)
        feats = torch.cat([feats, pad], dim=1)
    feats = feats[:, :T, :]
    feats_lengths = torch.tensor([T])   # full length -> no padding mask
    print(f"export: feats {tuple(feats.shape)} lens {feats_lengths.tolist()}")

    wrap = EncoderAvatar(asr.encoder).eval()
    with torch.no_grad():
        ref = wrap(feats, feats_lengths)
    print(f"  ref stacked shape={tuple(ref[0].shape)}")

    # NOTE: feats_lengths gets constant-folded during tracing (concrete value
    # 400 folds the length-dependent mask) -> the ONNX has a SINGLE input
    # `feats`. That's intended: fixed 1x400x560, no runtime lengths.
    torch.onnx.export(
        wrap, (feats,), dst,
        input_names=["feats"],
        output_names=["hidden_states"],
        opset_version=13, dynamic_axes=None,
    )
    print(f"  exported -> {dst}")
    return feats, feats_lengths, ref


def verify(dst, feats, feats_lengths, ref):
    import onnxruntime as ort
    sess = ort.InferenceSession(dst, providers=["CPUExecutionProvider"])
    in_names = [i.name for i in sess.get_inputs()]
    print(f"  verify: ONNX inputs={in_names}")
    feed = {in_names[0]: feats.numpy()}
    outs = sess.run(None, feed)
    hs = outs[0]   # [1, T, L, D]
    ref_np = ref[0].numpy()
    cos = float(np.dot(hs.reshape(-1), ref_np.reshape(-1)) /
                (np.linalg.norm(hs) * np.linalg.norm(ref_np) + 1e-12))
    max_abs = float(np.max(np.abs(hs - ref_np)))
    print(f"  verify: ONNX shape={hs.shape} cosine(vs torch)={cos:.6f} max_abs={max_abs:.6e}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="conversion/encoder_avatar.onnx")
    p.add_argument("--probe_only", action="store_true")
    a = p.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)

    s2t = build_s2t()
    asr, hidden_states, enc = probe(s2t)
    if hidden_states is None or len(hidden_states) == 0:
        print("ERROR: encoder returned no intermediate_outs — cannot build avatar encoder")
        sys.exit(1)
    if a.probe_only:
        return
    feats, fl, ref = export_encoder(asr, a.out)
    verify(a.out, feats, fl, ref)


if __name__ == "__main__":
    main()
