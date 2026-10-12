"""
Generate face-generator calibration data for the SPLIT net_encode + net_decode.

net_encode (single input):  calib_data/facegen_encode/calib_*.npy  [1,3,384,384]
net_decode (5 inputs):      calib_data/facegen_decode/<input>/k.npy  (multi-input
                            subdir layout per OE doc: one folder per input node,
                            samples matched by index k)

net_decode inputs (from net_encode on ref image) + mouth_param:
  f0 [1,3,384,384], f1 [1,16,192,192], f2 [1,32,96,96], f3 [1,64,48,48], mouth_param [1,32]

Ref-image preprocessing mirrors lite_avatar.py load_data():
  cv2.imread -> BGR2RGB -> resize(384,384) -> ToTensor -> Normalize([0.5],[0.5])
  i.e. float CHW in [-1,1], [1,3,384,384].

mouth_param: realistic values from the folded audio2mouth FP32 ONNX run on the
a2m calib windows (frame-level [1,32]).

Usage (from repo root):
    python conversion/scripts/gen_calib_facegen.py \
        --ref_dir data/preload/ref_frames --a2m_onnx conversion/model_1_folded.onnx \
        --a2m_calib conversion/calib_data/audio2mouth --out conversion/calib_data
"""
import os, sys, glob, argparse
import numpy as np
import torch
import cv2

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
os.chdir(REPO_ROOT)
sys.path.insert(0, REPO_ROOT)


def normalize_ref(path):
    img = cv2.cvtColor(cv2.imread(path)[:, :, 0:3], cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (384, 384), interpolation=cv2.INTER_LINEAR)
    t = img.astype(np.float32) / 255.0
    t = np.transpose(t, (2, 0, 1))
    t = 2.0 * t - 1.0
    return t[None, ...].astype(np.float32)       # [1,3,384,384]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ref_dir", default="data/preload/ref_frames")
    p.add_argument("--a2m_onnx", default="conversion/model_1_folded.onnx")
    p.add_argument("--a2m_calib", default="conversion/calib_data/audio2mouth")
    p.add_argument("--out", default="conversion/calib_data")
    p.add_argument("--n_img", type=int, default=30)
    a = p.parse_args()

    enc_out = os.path.join(a.out, "facegen_encode")
    dec_out = os.path.join(a.out, "facegen_decode")
    os.makedirs(enc_out, exist_ok=True)
    for sub in ["f0", "f1", "f2", "f3", "mouth_param"]:
        os.makedirs(os.path.join(dec_out, sub), exist_ok=True)

    refs = sorted(glob.glob(f"{a.ref_dir}/**/*.jpg", recursive=True))[: a.n_img]
    print(f"collected {len(refs)} ref images")
    if not refs:
        raise SystemExit("no ref images")

    # 1. encode calib (normalized images) + run net_encode for features
    enc = torch.jit.load("data/preload/net_encode.pt").eval()
    feats_all = []   # list of [f0,f1,f2,f3]
    for i, r in enumerate(refs):
        x = normalize_ref(r)
        np.save(os.path.join(enc_out, f"calib_{i:03d}.npy"), x)
        with torch.no_grad():
            fs = enc(torch.from_numpy(x))     # list[4]
        feats_all.append([f.numpy() for f in fs])
    print(f"saved {len(refs)} encode calib + ran net_encode for features")

    # 2. mouth params from a2m FP32
    import onnxruntime as ort
    sess = ort.InferenceSession(a.a2m_onnx, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    out_name = "output" if "output" in [o.name for o in sess.get_outputs()] else sess.get_outputs()[0].name
    wins = sorted(glob.glob(f"{a.a2m_calib}/*.npy"))
    params = []
    for w in wins:
        x = np.load(w).astype(np.float32)
        y = sess.run([out_name], {in_name: x})[0]   # [1,30,32]
        for t in range(y.shape[1]):
            params.append(y[0, t:t + 1, :].astype(np.float32))   # [1,32]
    print(f"collected {len(params)} mouth-param frames")

    # 3. net_decode multi-input calib: matched-index samples across 5 subdirs
    n_pair = len(refs)
    for i in range(n_pair):
        fs = feats_all[i]
        par = params[i % len(params)]
        np.save(os.path.join(dec_out, "f0", f"{i}.npy"), fs[0])
        np.save(os.path.join(dec_out, "f1", f"{i}.npy"), fs[1])
        np.save(os.path.join(dec_out, "f2", f"{i}.npy"), fs[2])
        np.save(os.path.join(dec_out, "f3", f"{i}.npy"), fs[3])
        np.save(os.path.join(dec_out, "mouth_param", f"{i}.npy"), par)
    print(f"saved {n_pair} multi-input decode samples (5 subdirs) -> {dec_out}")
    print("  subdirs:", sorted(os.listdir(dec_out)))


if __name__ == "__main__":
    main()
