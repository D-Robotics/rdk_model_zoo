#!/usr/bin/env python3
"""Generate net_decode calibration with REAL animated mouth params (runtime-exact).

Runs the SAME board pipeline as the validated source (encoder_avatar.hbm +
audio2mouth.hbm for real butter-smoothed mouth params; net_encode.hbm for ref
features) and dumps a multi-input calib set where mouth_param spans the real
animation instead of neutral/ref values.  Output layout matches hb_compile's
multi-input convention: <out>/{f0,f1,f2,f3,mouth_param}/k.npy.

The helpers below are the validated wrapper/dequant/frontend/param-loop code
of the RDK adaptation source (paths parameterized; arithmetic unchanged).

Run ON the board, from a workspace holding the upstream lite-avatar checkout
(funasr_local importable) and the prepared HBMs:

  python3 gen_real_calib.py --hbm-dir /path/to/hbm --data /path/to/preload \
      --mvn /path/to/am.mvn --wav /path/to/asr_example.wav \
      --out calib_data/facegen_decode_real --upstream /path/to/lite-avatar
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from scipy import signal

HB_DTYPE_MAP = {
    "U8": np.uint8, "S8": np.int8, "F32": np.float32, "F16": np.float16,
    "U16": np.uint16, "S16": np.int16, "U32": np.uint32, "S32": np.int32,
    "BOOL8": np.bool_,
}


def dequant(arr, qinfo):
    """Dequantize a board output to float32 via QuantParams (per-tensor / per-channel SCALE)."""
    if qinfo is None or qinfo.quant_type.name != "SCALE":
        return arr.astype(np.float32)
    scale = np.asarray(qinfo.scale, dtype=np.float32)
    zp = np.asarray(qinfo.zero_point, dtype=np.float32)
    axis = qinfo.axis
    arr = arr.astype(np.float32)
    if scale.size == 1:                       # per-tensor
        return (arr - zp) * scale
    shape = [1] * arr.ndim                    # per-channel along `axis`
    shape[axis] = arr.shape[axis]
    return (arr - zp) * scale.reshape(shape)


class HBM:
    """Thin wrapper: load one HBM, run a feed dict, dequantize outputs."""
    def __init__(self, path):
        from hbm_runtime import HB_HBMRuntime
        self.m = HB_HBMRuntime(path)
        self.name = self.m.model_names[0]
        self.ins = self.m.input_names[self.name]
        self.in_dtypes = self.m.input_dtypes[self.name]
        self.outs = self.m.output_names[self.name]
        self.oq = self.m.output_quants[self.name]
        print(f"  [HBM] {os.path.basename(path)}: ins={self.ins} outs={self.outs}")

    def run(self, feed):
        # feed: {input_name: float ndarray}; cast to expected dtype, contiguous
        cast = {}
        for nm, arr in feed.items():
            dt = HB_DTYPE_MAP.get(self.in_dtypes[nm].name, np.float32)
            cast[nm] = np.ascontiguousarray(arr.astype(dt, copy=False))
        res = self.m.run(cast[self.ins[0]]) if len(self.ins) == 1 else self.m.run(cast)
        out = res[self.name]
        deq = {}
        for oname in self.outs:
            arr = np.asarray(out[oname]) if oname in out else np.asarray(list(out.values())[0])
            deq[oname] = dequant(arr, self.oq.get(oname))
        return deq


def wav_to_feats(audio, sr, am_mvn):
    """16k mono audio np -> post-CMVN LFR featuremap [T,560] (upstream frontend)."""
    from funasr_local.models.frontend.wav_frontend import WavFrontend
    if sr != 16000:
        # scipy polyphase resample (no librosa needed)
        from fractions import Fraction
        frac = Fraction(16000, sr).limit_denominator(10000)
        audio = signal.resample_poly(audio, frac.numerator, frac.denominator)
    audio = audio.astype(np.float32)
    if audio.ndim > 1:
        audio = audio[:, 0]
    fe = WavFrontend(cmvn_file=am_mvn, fs=16000, window="hamming", n_mels=80,
                     frame_length=25, frame_shift=10, lfr_m=7, lfr_n=6,
                     dither=0.0, snip_edges=True)
    speech = torch.from_numpy(audio).unsqueeze(0)
    feats, _ = fe(speech, torch.tensor([len(audio)]))
    return feats[0].numpy()        # [T, 560]


def audio_to_au_data(audio, sr, frame_cnt, enc, am_mvn):
    """wav -> feats -> encoder_avatar.hbm -> hidden_states -> interp -> au_data [frame_cnt,50,512]."""
    feats = wav_to_feats(audio, sr, am_mvn)          # [T,560]
    T = feats.shape[0]
    T_real = min(T, 400)
    if T >= 400:
        f400 = feats[:400]
    else:                                              # pad to fixed 400
        f400 = np.pad(feats, ((0, 400 - T), (0, 0)), mode="constant")
    f400 = f400[None, ...].astype(np.float32)          # [1,400,560]
    hs = enc.run({"feats": f400})["hidden_states"][0]   # [400,50,512] (dequanted)
    hs = hs[:T_real]                                   # drop padding frames
    # interpolate time axis T_real -> frame_cnt, per (layer, dim).
    t = torch.from_numpy(hs).float().permute(1, 2, 0)          # [50,512,T_real]
    t = F.interpolate(t, size=frame_cnt, mode="linear", align_corners=True)
    au = t.permute(2, 0, 1).numpy().astype(np.float32)         # [frame_cnt,50,512]
    return au


def _butter(data, order=4, cutoff=10, fs=30):
    wn = 2 * cutoff / fs
    b, a = signal.butter(order, wn, "lowpass", analog=False)
    return signal.filtfilt(b, a, data, padtype=None, axis=0).astype(np.float32)


def au_to_params(au_data, a2m):
    """au_data [frame_cnt,50,512] -> mouth params [frame_cnt,32] (1s window, frag crossfade, butterworth)."""
    frame_cnt = au_data.shape[0]
    audio_length = frame_cnt / 30.0
    interval = 1.0
    frag = int(interval * 30 / 5 + 0.5)          # 6
    win = int(30 * interval)                      # 30
    param_res = []                                # list of [32] np arrays
    start_time, end_time, is_end = 0.0, interval, False
    while True:
        if end_time >= audio_length:
            is_end = True
            start_time = max(audio_length - interval, 0.0)
            end_time = audio_length
        start_frame = int(start_time * 30)
        end_frame = start_frame + win
        input_au = au_data[start_frame:end_frame]                 # [<=30,50,512]
        if input_au.shape[0] < win:
            input_au = np.pad(input_au, ((0, win - input_au.shape[0]), (0, 0), (0, 0)))
        input_au = input_au[None].astype(np.float32)             # [1,30,50,512]
        out = a2m.run({"input_au": input_au})["output"][0]       # [30,32]

        if start_time == 0.0:
            rng = range(win) if is_end else range(win - frag)
            for tt in rng:
                param_res.append(out[tt].copy())
        elif not is_end:
            for tt in range(frag, win - frag):
                fid = start_frame + tt
                if fid < len(param_res):
                    s = min((len(param_res) - fid) / frag, 1.0)
                    param_res[fid] = (1 - s) * out[tt] + s * param_res[fid]
                else:
                    param_res.append(out[tt].copy())
        else:
            for tt in range(frag, win):
                fid = start_frame + tt
                if fid < len(param_res):
                    s = min((len(param_res) - fid) / frag, 1.0)
                    param_res[fid] = (1 - s) * out[tt] + s * param_res[fid]
                else:
                    param_res.append(out[tt].copy())

        start_time = end_time - (frag / 10)                       # overlap 0.6s
        end_time = start_time + interval
        if is_end:
            break

    param_res = np.stack(param_res, axis=0).astype(np.float32)   # [N,32]
    # align to frame_cnt (trim or pad with last)
    if len(param_res) >= frame_cnt:
        param_res = param_res[:frame_cnt]
    else:
        pad = np.repeat(param_res[-1:], frame_cnt - len(param_res), axis=0)
        param_res = np.concatenate([param_res, pad], axis=0)
    param_res = _butter(param_res)                               # butterworth smooth per-param
    return param_res


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hbm-dir", required=True, help="directory with the 3 pre-quantized HBMs")
    parser.add_argument("--data", required=True, help="avatar preload dir (ref_frames/)")
    parser.add_argument("--mvn", required=True, help="paraformer am.mvn")
    parser.add_argument("--wav", required=True, help="example wav (16 kHz)")
    parser.add_argument("--out", required=True, help="output calib dir")
    parser.add_argument("--upstream", required=True,
                        help="upstream lite-avatar checkout root (funasr_local importable)")
    parser.add_argument("--n-ref", type=int, default=150)
    args = parser.parse_args()
    sys.path.insert(0, args.upstream)

    import soundfile as sf
    print("loading HBMs ...", flush=True)
    enc = HBM(f"{args.hbm_dir}/encoder_avatar.hbm")
    a2m = HBM(f"{args.hbm_dir}/audio2mouth.hbm")
    fgenc = HBM(f"{args.hbm_dir}/net_encode.hbm")

    # 1) real animated mouth params (butter-smoothed, exactly what net_decode gets)
    audio, sr = sf.read(args.wav)
    audio = np.asarray(audio, dtype=np.float32)
    frame_cnt = int(len(audio) / 16000 * 30)
    au = audio_to_au_data(audio, sr, frame_cnt, enc, args.mvn)
    params = au_to_params(au, a2m)                       # [frame_cnt,32]
    print(f"real params: {params.shape}  range=[{params.min():.3f},{params.max():.3f}] "
          f"std={params.std():.3f}", flush=True)
    np.save(os.path.join(args.out, "..", "real_params.npy"), params)

    # 2) ref features via net_encode.hbm (identity features, same as runtime)
    import cv2
    refs = sorted(glob.glob(f"{args.data}/ref_frames/ref_*.jpg"))[: args.n_ref]
    feats = []
    for r in refs:
        img = cv2.cvtColor(cv2.imread(r)[:, :, 0:3], cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (384, 384), interpolation=cv2.INTER_LINEAR)
        t = img.astype(np.float32) / 255.0
        t = np.transpose(t, (2, 0, 1))
        t = 2.0 * t - 1.0
        x = t[None, ...].astype(np.float32)                # [1,3,384,384]
        o = fgenc.run({"image": x})
        feats.append((o["f0"], o["f1"], o["f2"], o["f3"]))
    print(f"ref features: {len(feats)}", flush=True)

    # 3) write multi-input calib, mouth_param cycling over the animation
    for sub in ["f0", "f1", "f2", "f3", "mouth_param"]:
        os.makedirs(f"{args.out}/{sub}", exist_ok=True)
    n = len(feats)
    T = params.shape[0]
    for i in range(n):
        f0, f1, f2, f3 = feats[i]
        par = params[i % T][None, :].astype(np.float32)     # [1,32]
        np.save(f"{args.out}/f0/{i}.npy", f0)
        np.save(f"{args.out}/f1/{i}.npy", f1)
        np.save(f"{args.out}/f2/{i}.npy", f2)
        np.save(f"{args.out}/f3/{i}.npy", f3)
        np.save(f"{args.out}/mouth_param/{i}.npy", par)
    print(f"saved {n} samples -> {args.out}")
    for sub in ["f0", "f1", "f2", "f3", "mouth_param"]:
        files = sorted(glob.glob(f"{args.out}/{sub}/*.npy"))
        print(f"  {sub}: {len(files)} files", flush=True)


if __name__ == "__main__":
    main()
