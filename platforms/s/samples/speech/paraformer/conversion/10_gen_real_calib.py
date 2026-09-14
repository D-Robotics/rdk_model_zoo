"""
Step 10: Generate REAL calibration data for predictor + decoder.

CRITICAL: Do NOT use random data for calibration. Real-distribution data is
required or INT16 quantization will fail catastrophically (CER → 100%).

This script runs the FP32 pipeline on the encoder calibration set (produced by
09_gen_calib_features.py), captures intermediate values, and stores them as
per-tensor calibration directories:

    real_calib/
    ├── encoder_after_norm_Add_1_output_0/    # 50 × [1,400,512]
    ├── predictor_Concat_5_output_0/          # 50 × [1,401,512]
    ├── predictor_Add_output_0/               # 50 × [1,401]
    ├── shape_8609/                           # 50 × [1,100,512]  (pre_acoustic_embeds)
    ├── token_num/                            # 50 × [1] int32
    └── bias_embed/                           # 50 × [1,1,512] all-zero

Usage:
    python 10_gen_real_calib.py \
        --enc encoder_only.onnx \
        --pred predictor_only.onnx \
        --calib_feats ./calib_data/speech \
        --out ./real_calib
"""
import os, sys, glob, argparse
import numpy as np
import onnxruntime as ort

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cif_numpy import cif_numpy


def main(enc_path, pred_path, calib_dir, out_base):
    enc = ort.InferenceSession(enc_path, providers=["CPUExecutionProvider"])
    pred = ort.InferenceSession(pred_path, providers=["CPUExecutionProvider"])

    subdirs = {
        "encoder_after_norm_Add_1_output_0": None,
        "predictor_Concat_5_output_0": None,
        "predictor_Add_output_0": None,
        "shape_8609": None,
        "token_num": None,
        "bias_embed": None,
    }
    for name in subdirs:
        d = os.path.join(out_base, name)
        os.makedirs(d, exist_ok=True)
        subdirs[name] = d

    feat_files = sorted(glob.glob(f"{calib_dir}/*.npy"))
    print(f"generating real calib from {len(feat_files)} FP32 pipeline runs")

    for i, f in enumerate(feat_files):
        speech = np.load(f)
        if speech.shape != (1, 400, 560):
            print(f"  skip {f}: shape={speech.shape}"); continue

        enc_out = enc.run(None, {"speech": speech})[0]
        pr_out = pred.run(None, {"/encoder/after_norm/Add_1_output_0": enc_out})
        pr_names = [o.name for o in pred.get_outputs()]
        pr = dict(zip(pr_names, pr_out))
        alphas = pr["/predictor/Add_output_0"]
        concat5 = pr["/predictor/Concat_5_output_0"]
        # For calibration we use full-length (no real_T mask): captures wider dist
        frame_fires, token_num = cif_numpy(alphas, concat5, real_T=None)
        bias = np.zeros((1, 1, 512), dtype=np.float32)

        np.save(f"{subdirs['encoder_after_norm_Add_1_output_0']}/{i:03d}.npy", enc_out)
        np.save(f"{subdirs['predictor_Concat_5_output_0']}/{i:03d}.npy", concat5)
        np.save(f"{subdirs['predictor_Add_output_0']}/{i:03d}.npy", alphas)
        np.save(f"{subdirs['shape_8609']}/{i:03d}.npy", frame_fires)
        np.save(f"{subdirs['token_num']}/{i:03d}.npy", token_num)
        np.save(f"{subdirs['bias_embed']}/{i:03d}.npy", bias)
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}] enc_out {enc_out.shape}[{enc_out.min():.2f},{enc_out.max():.2f}]  "
                  f"alphas sum={alphas.sum():.1f}  token_num={int(token_num[0])}")

    print(f"\n== real calibration data ready in {out_base} ==")
    for name, d in subdirs.items():
        arr = np.load(sorted(glob.glob(f"{d}/*.npy"))[0])
        print(f"  {name}: shape={arr.shape} dtype={arr.dtype} range=[{arr.min():.3f},{arr.max():.3f}]")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--enc",         default="./out/encoder_only.onnx")
    p.add_argument("--pred",        default="./out/predictor_only.onnx")
    p.add_argument("--calib_feats", default="./calib_data/speech")
    p.add_argument("--out",         default="./real_calib")
    args = p.parse_args()
    main(args.enc, args.pred, args.calib_feats, args.out)
