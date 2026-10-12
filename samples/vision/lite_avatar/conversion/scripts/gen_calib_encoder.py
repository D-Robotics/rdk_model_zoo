"""
Generate encoder-avatar calibration data: post-CMVN featuremaps [1,400,560].

Runs the funasr_local paraformer frontend (fbank+LFR+CMVN) on 16kHz wavs,
pads/truncates to 400 frames, saves [1,400,560] float32 .npy — matching the
model-zoo encoder calib format (09_gen_calib_features.py) but using
funasr_local (no upstream funasr needed).

Usage (from repo root, weights/ populated):
    python conversion/scripts/gen_calib_encoder.py \
        --wav_dir funasr_local/runtime/triton_gpu/client/test_wavs \
        --out conversion/calib_data/encoder_avatar --n 50
"""
import os, sys, glob, argparse
import numpy as np
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
os.chdir(REPO_ROOT)
sys.path.insert(0, REPO_ROOT)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--wav_dir", default="funasr_local/runtime/triton_gpu/client/test_wavs")
    p.add_argument("--paraformer_example_wav", default="weights/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/example")
    p.add_argument("--out", default="conversion/calib_data/encoder_avatar")
    p.add_argument("--n", type=int, default=50)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)

    from extract_paraformer_feature import Speech2Text, speech2text_kwargs
    s2t = Speech2Text(**speech2text_kwargs)
    asr = s2t.asr_model
    asr.eval()
    import soundfile as sf

    wavs = sorted(glob.glob(f"{a.wav_dir}/**/*.wav", recursive=True))
    wavs += sorted(glob.glob(f"{a.paraformer_example_wav}/**/*.wav", recursive=True))
    seen = set(); uniq = []
    for w in wavs:
        if w not in seen:
            seen.add(w); uniq.append(w)
    wavs = uniq[: a.n]
    print(f"collected {len(wavs)} wavs")

    saved = 0
    for i, w in enumerate(wavs):
        audio, sr = sf.read(w)
        if audio.ndim > 1:
            audio = audio[:, 0]
        audio = audio.astype(np.float32)
        if sr != 16000:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        speech = torch.from_numpy(audio).unsqueeze(0)
        speech_lengths = torch.tensor([speech.shape[1]])
        with torch.no_grad():
            feats, feats_lengths = asr._extract_feats(speech, speech_lengths)
        feats_np = feats[0].detach().cpu().numpy()  # [T, 560]
        T = feats_np.shape[0]
        if T >= 400:
            f400 = feats_np[:400]
        else:
            f400 = np.pad(feats_np, ((0, 400 - T), (0, 0)), mode="constant")
        f400 = f400[None, ...].astype(np.float32)  # [1,400,560]
        np.save(os.path.join(a.out, f"calib_{saved:03d}.npy"), f400)
        saved += 1
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(wavs)}] {os.path.basename(w)}: T={T} -> 400")
    print(f"\nSaved {saved} encoder calib samples [1,400,560] -> {a.out}")


if __name__ == "__main__":
    main()
