"""
Step 01: Re-export FunASR Paraformer ONNX with CIF monkey-patch.

Patches applied:
  - cif_v1_export: force max_label_len=100 (was: floor(alphas.sum(-1)).max())
    → makes output frame_fires shape fully static [B, 100, hidden_size]
  - cif_v1_export: rewrite .unsqueeze(-1) as .reshape(-1, 1)
    → avoids hbdk4 type_inf crash on Unsqueeze nodes downstream of GatherND
  - export_backbone_forward: force pre_token_length=100
    → decoder tgt_mask stays fixed size, matches acoustic_embeds

Usage:
    # Adjust MODEL_DIR to your FunASR-downloaded paraformer directory
    python 01_reexport_fixed_shape.py

Outputs:
    <MODEL_DIR>/model.onnx      - Full FP32 model with patched CIF
    <MODEL_DIR>/model_eb.onnx   - Contextual embedder (not used in this pipeline)
"""
import os
import torch

MAX_LABEL_LEN = 100
MODEL_DIR = "./models/paraformer"  # ← adjust to your FunASR model dir


# ============================================================
# Patch 1: cif_v1_export — static max_label_len + Reshape swap
# ============================================================
def patched_cif_v1_export(hidden, alphas, threshold: float):
    device = hidden.device
    dtype = hidden.dtype
    batch_size, len_time, hidden_size = hidden.size()

    frames = torch.zeros(batch_size, len_time, hidden_size, dtype=dtype, device=device)
    fires = torch.zeros(batch_size, len_time, dtype=dtype, device=device)

    prefix_sum = torch.cumsum(alphas, dim=1, dtype=torch.float64).to(torch.float32)
    prefix_sum_floor = torch.floor(prefix_sum)
    dislocation_prefix_sum = torch.roll(prefix_sum, 1, dims=1)
    dislocation_prefix_sum_floor = torch.floor(dislocation_prefix_sum)
    dislocation_prefix_sum_floor[:, 0] = 0
    dislocation_diff = prefix_sum_floor - dislocation_prefix_sum_floor

    fire_idxs = dislocation_diff > 0
    fires = fires.masked_fill(fire_idxs, 1.0)
    fires = fires + prefix_sum - prefix_sum_floor

    # Static branch (no data-dependent zero-fire early-return).
    # All `.unsqueeze(...)` calls rewritten as `.reshape(...)` to avoid the
    # ONNX Unsqueeze op that hbdk4's type_inf pass crashes on.
    alphas_r = alphas.reshape(batch_size, len_time, 1)
    prefix_sum_hidden = torch.cumsum(
        alphas_r.repeat((1, 1, hidden_size)) * hidden, dim=1
    )
    frames = prefix_sum_hidden[fire_idxs]
    shift_frames = torch.roll(frames, 1, dims=0)

    batch_len = fire_idxs.sum(1)
    batch_idxs = torch.cumsum(batch_len, dim=0)
    shift_batch_idxs = torch.roll(batch_idxs, 1, dims=0)
    shift_batch_idxs[0] = 0
    shift_frames[shift_batch_idxs] = 0

    remains = fires - torch.floor(fires)
    remains_sel = remains[fire_idxs]
    hidden_sel = hidden[fire_idxs]
    remains_sel_2d = remains_sel.reshape(-1, 1)   # instead of unsqueeze(-1)
    remain_frames = remains_sel_2d.repeat((1, hidden_size)) * hidden_sel
    shift_remain_frames = torch.roll(remain_frames, 1, dims=0)
    shift_remain_frames[shift_batch_idxs] = 0

    frames = frames - shift_frames + shift_remain_frames - remain_frames

    max_label_len: int = MAX_LABEL_LEN

    frame_fires = torch.zeros(batch_size, max_label_len, hidden_size, dtype=dtype, device=device)
    indices = torch.arange(max_label_len, device=device).expand(batch_size, -1)
    batch_len_clamped = torch.clamp(batch_len, max=max_label_len)
    batch_len_2d = batch_len_clamped.reshape(batch_size, 1)  # instead of unsqueeze(1)
    frame_fires_idxs = indices < batch_len_2d
    num_slots = int(frame_fires_idxs.sum().item())
    frame_fires[frame_fires_idxs] = frames[:num_slots]
    return frame_fires, fires


# ============================================================
# Patch 2: export_backbone_forward — force pre_token_length=100
# ============================================================
def patched_export_backbone_forward(self, speech, speech_lengths, bias_embed):
    batch = {"speech": speech, "speech_lengths": speech_lengths}
    enc, enc_len = self.encoder(**batch)
    mask = self.make_pad_mask(enc_len)[:, None, :]
    pre_acoustic_embeds, _pre_token_length, _, _ = self.predictor(enc, mask)
    B = pre_acoustic_embeds.shape[0]
    pre_token_length_fixed = torch.full(
        (B,), MAX_LABEL_LEN, dtype=torch.int32, device=pre_acoustic_embeds.device
    )
    decoder_out, _ = self.decoder(
        enc, enc_len, pre_acoustic_embeds, pre_token_length_fixed, bias_embed
    )
    decoder_out = torch.log_softmax(decoder_out, dim=-1)
    return decoder_out, pre_token_length_fixed


if __name__ == "__main__":
    # Apply patches BEFORE FunASR imports its export path
    import funasr.models.paraformer.cif_predictor as cifp
    cifp._original_cif_v1_export = cifp.cif_v1_export
    cifp.cif_v1_export = torch.jit.script(patched_cif_v1_export)
    print(f"[patch] cif_v1_export: max_label_len fixed to {MAX_LABEL_LEN}")

    try:
        from funasr.models.paraformer import cif_predictor as _cp
        _cp.cif_v1_export = cifp.cif_v1_export
    except Exception:
        pass

    import funasr.models.contextual_paraformer.export_meta as cp_em
    cp_em.export_backbone_forward = patched_export_backbone_forward
    print(f"[patch] export_backbone_forward: pre_token_length forced to {MAX_LABEL_LEN}")

    from funasr import AutoModel
    os.chdir(MODEL_DIR)
    m = AutoModel(model=".", disable_update=True)

    for f in ("model.onnx", "model_eb.onnx"):
        if os.path.exists(f):
            os.remove(f)
            print(f"removed old {f}")

    ret = m.export(quantize=False, opset_version=15)
    print(f"export done: {ret}")
    print(f"produced: {[f for f in os.listdir('.') if f.endswith('.onnx')]}")
