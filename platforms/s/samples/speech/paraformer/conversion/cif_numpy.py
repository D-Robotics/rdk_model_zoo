"""
Standalone CIF (Continuous Integrate-and-Fire) numpy implementation.

Used at inference time between predictor and decoder.
Also used at calibration data generation time (see 10_gen_real_calib.py).

CRITICAL: `alphas[:, real_T:] = 0` before CIF, otherwise padding-region
spurious fires cause CER to jump from ~5% to ~44%.
"""
import numpy as np

MAX_LABEL_LEN = 100


def cif_numpy(alphas, concat5, real_T=None):
    """
    Args:
        alphas:  [1, 401] fp32 — predictor `/predictor/Add_output_0`
        concat5: [1, 401, 512] fp32 — predictor `/predictor/Concat_5_output_0`
        real_T:  int or None — real LFR frame count (pre-padding).
                 If None, no masking (only use for calibration data generation).
                 At inference, ALWAYS pass real_T.

    Returns:
        frame_fires: [1, 100, 512] fp32 — pre_acoustic_embeds (decoder query)
        token_num:   [1] int32 — number of predicted tokens
    """
    alphas = alphas.copy()
    if real_T is not None and real_T < alphas.shape[1]:
        alphas[:, real_T:] = 0.0

    B, T = alphas.shape
    H = concat5.shape[-1]

    prefix_sum = np.cumsum(alphas.astype(np.float64), axis=1).astype(np.float32)
    prefix_sum_floor = np.floor(prefix_sum)
    disl_ps_floor = np.floor(np.roll(prefix_sum, 1, axis=1))
    disl_ps_floor[:, 0] = 0
    fire_idxs = (prefix_sum_floor - disl_ps_floor) > 0

    fires = np.zeros_like(prefix_sum)
    fires[fire_idxs] = 1.0
    fires = fires + prefix_sum - prefix_sum_floor

    prefix_sum_hidden = np.cumsum(
        alphas[..., None].astype(np.float64) * concat5.astype(np.float64),
        axis=1,
    ).astype(np.float32)
    frames = prefix_sum_hidden[fire_idxs]
    shift_frames = np.roll(frames, 1, axis=0)

    batch_len = fire_idxs.sum(axis=1)
    batch_idxs = np.cumsum(batch_len)
    shift_batch_idxs = np.roll(batch_idxs, 1)
    shift_batch_idxs[0] = 0
    shift_frames[shift_batch_idxs] = 0

    remains = fires - np.floor(fires)
    remain_frames = remains[fire_idxs][:, None] * concat5[fire_idxs]
    shift_remain_frames = np.roll(remain_frames, 1, axis=0)
    shift_remain_frames[shift_batch_idxs] = 0

    frames = frames - shift_frames + shift_remain_frames - remain_frames

    frame_fires = np.zeros((B, MAX_LABEL_LEN, H), dtype=np.float32)
    indices = np.arange(MAX_LABEL_LEN)[None, :]
    batch_len_clamped = np.clip(batch_len, None, MAX_LABEL_LEN)
    slot_mask = indices < batch_len_clamped[:, None]
    num_slots = int(slot_mask.sum())
    frame_fires[slot_mask] = frames[:num_slots]

    return frame_fires, batch_len_clamped.astype(np.int32)
