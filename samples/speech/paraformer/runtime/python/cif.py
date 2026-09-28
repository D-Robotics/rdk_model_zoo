"""Fixed-shape Paraformer CPU bridge, shared by inference and calibration.

Numerical integration follows the S source at 380e1a2. This is not CTC:
one acoustic embedding is emitted at each integer cumulative-weight crossing.
"""

import numbers

import numpy as np

MAX_LABEL_LEN = 100


def cif_numpy(alphas, concat5, *, real_T):
    """Return float32 [1,100,512] embeddings and int32 [1] token count.

    Inputs must be finite float32 [1,401] weights and [1,401,512] hidden
    states. Weights must be nonnegative. Supply real_T in [0,400] to mask
    padding at inference, or explicitly None for source-compatible unmasked
    calibration. More than 100 emitted embeddings are truncated. Inputs
    are never modified; outputs own their memory. No-fire input returns
    zeros. This implements the source's one-fire-per-frame algorithm, not
    arbitrary multi-fire integration for weights exceeding one.
    """
    for name, value, shape in (
        ("alphas", alphas, (1, 401)),
        ("concat5", concat5, (1, 401, 512)),
    ):
        if not isinstance(value, np.ndarray) or value.dtype != np.dtype("float32"):
            raise TypeError(f"{name} must be a float32 ndarray")
        if value.shape != shape:
            raise ValueError(f"{name} must have shape {shape}; got {value.shape}")
        if not np.isfinite(value).all():
            raise ValueError(f"{name} must contain only finite values")
    if np.any(alphas < 0):
        raise ValueError("alphas must be nonnegative")
    if real_T is not None:
        if isinstance(real_T, (bool, np.bool_)) or not isinstance(
            real_T, numbers.Integral
        ):
            raise TypeError(
                "real_T must be an integer or explicit None for calibration"
            )
        if not 0 <= real_T <= 400:
            raise ValueError("real_T must be between 0 and 400")
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
    if frames.shape[0] == 0:
        return (
            np.zeros((1, MAX_LABEL_LEN, 512), dtype=np.float32),
            np.zeros(1, dtype=np.int32),
        )
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
