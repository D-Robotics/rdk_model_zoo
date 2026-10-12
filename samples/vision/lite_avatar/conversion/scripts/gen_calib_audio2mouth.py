"""
Generate audio2mouth calibration data (input_au windows [1,30,50,512]).

Runs the lite-avatar paraformer feature extractor (funasr_local) on a set of
16kHz wavs to obtain the 50-layer encoder hidden states (intermediate_outs),
linearly interpolated to the video frame count — exactly the `au_data` that
audio2mouth_cpu.py feeds as `input_au`. Slices into 30-frame windows and saves
each as a .npy in the calib dir for hb_compile.

Reuses extract_paraformer_feature.extract_para_feature verbatim for fidelity.

Usage (run from lite-avatar repo root, with weights/ populated):
    python conversion/scripts/gen_calib_audio2mouth.py \
        --wav_dir funasr_local/runtime/triton_gpu/client/test_wavs \
        --paraformer_example_wav weights/speech_paraformer-*/example \
        --out conversion/calib_data/audio2mouth --n_windows 60
"""
import os, sys, glob, argparse, pathlib
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
os.chdir(REPO_ROOT)
sys.path.insert(0, REPO_ROOT)


def collect_wavs(wav_dir, example_dir, n_wavs=20):
    wavs = sorted(glob.glob(f"{wav_dir}/**/*.wav", recursive=True))
    if example_dir:
        wavs += sorted(glob.glob(f"{example_dir}/**/*.wav", recursive=True))
    # de-dup preserving order
    seen = set(); out = []
    for w in wavs:
        if w not in seen:
            seen.add(w); out.append(w)
    return out[:n_wavs]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--wav_dir", default="funasr_local/runtime/triton_gpu/client/test_wavs")
    p.add_argument("--paraformer_example_wav", default=None,
                   help="dir with paraformer example wavs")
    p.add_argument("--out", default="conversion/calib_data/audio2mouth")
    p.add_argument("--n_wavs", type=int, default=20)
    p.add_argument("--n_windows", type=int, default=60)
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # reuse lite-avatar's own feature extractor (loads paraformer at import)
    from extract_paraformer_feature import extract_para_feature
    import soundfile as sf

    wavs = collect_wavs(args.wav_dir, args.paraformer_example_wav, args.n_wavs)
    print(f"collected {len(wavs)} wavs: {[os.path.basename(w) for w in wavs]}")
    if not wavs:
        raise SystemExit("no wavs found for calibration")

    windows = []
    for w in wavs:
        audio, sr = sf.read(w)
        if audio.ndim > 1:
            audio = audio[:, 0]
        audio = audio.astype(np.float32)
        if sr != 16000:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            sr = 16000
        frame_cnt = int(len(audio) / 16000 * 30)
        if frame_cnt < 30:
            print(f"  skip {os.path.basename(w)}: too short ({frame_cnt} frames)")
            continue
        au_data = extract_para_feature(audio, frame_cnt)   # [frame_cnt, 50, 512]
        print(f"  {os.path.basename(w)}: au_data {au_data.shape} from {frame_cnt} frames")
        # slice non-overlapping 30-frame windows
        for s in range(0, au_data.shape[0] - 30 + 1, 30):
            win = au_data[s:s+30]              # [30, 50, 512]
            win = win[np.newaxis, ...].astype(np.float32)  # [1,30,50,512]
            windows.append(win)
        if len(windows) >= args.n_windows:
            break

    if len(windows) < args.n_windows:
        print(f"only {len(windows)} windows collected (< {args.n_windows}); using all")
    if not windows:
        raise SystemExit("no 30-frame windows extracted")

    for i, win in enumerate(windows):
        np.save(os.path.join(args.out, f"calib_{i:03d}.npy"), win)
    print(f"\nSaved {len(windows)} calibration windows [1,30,50,512] → {args.out}")


if __name__ == "__main__":
    main()
