"""
Step 09: Extract fbank+LFR features from a wav directory (encoder calibration data).

Uses FunASR's built-in WavFrontend. Pads/truncates each utterance to 400 frames
(matches the encoder's fixed shape [1, 400, 560]).

Usage:
    python 09_gen_calib_features.py <wav_dir> <out_dir> [--n 50]

Example:
    python 09_gen_calib_features.py ./aishell_dev/wav/dev ./calib_data/speech --n 50
"""
import os, sys, glob, argparse
from pathlib import Path
import numpy as np
import torch
import soundfile as sf


def main(wav_dir, out_dir, n_samples, model_dir):
    os.makedirs(out_dir, exist_ok=True)
    from funasr import AutoModel
    model = AutoModel(model=model_dir, disable_update=True)
    frontend = model.kwargs.get("frontend")
    assert frontend is not None, "FunASR frontend not found; check model_dir"

    wavs = sorted(glob.glob(f"{wav_dir}/**/*.wav", recursive=True))[:n_samples]
    print(f"processing {len(wavs)} wavs → {out_dir}")

    for i, wav_path in enumerate(wavs):
        wav_np, sr = sf.read(wav_path)
        assert sr == 16000, f"expected 16 kHz, got {sr} in {wav_path}"
        wav_t = torch.from_numpy(wav_np).float().unsqueeze(0)
        feats, _ = frontend(wav_t, torch.tensor([wav_t.shape[1]]))
        feats_np = feats[0].detach().cpu().numpy()  # [T, 560]
        T = feats_np.shape[0]
        if T >= 400:
            feats_400 = feats_np[:400]
        else:
            feats_400 = np.pad(feats_np, ((0, 400 - T), (0, 0)), mode="constant")
        feats_400 = feats_400[None, ...].astype(np.float32)  # [1, 400, 560]
        out_path = f"{out_dir}/calib_{i:03d}.npy"
        np.save(out_path, feats_400)
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(wavs)}] {Path(wav_path).stem}: real_T={T} → 400")

    print(f"\nSaved {len(wavs)} calibration samples to {out_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("wav_dir", help="Directory of 16kHz wav files (recursively scanned)")
    p.add_argument("out_dir", default="./calib_data/speech")
    p.add_argument("--n", type=int, default=50, help="Number of samples to extract")
    p.add_argument("--model_dir", default="./models/paraformer", help="FunASR paraformer dir")
    args = p.parse_args()
    main(args.wav_dir, args.out_dir, args.n, args.model_dir)
