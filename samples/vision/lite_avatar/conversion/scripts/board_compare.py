"""
Host-side cosine comparison: FP32 ONNX vs board HBM outputs (dumped by board_infer.py).

Loads the board's dequantized output .npy files (named <idx>_<out_name>.npy),
runs the FP32 ONNX on the same calib inputs (onnxruntime CPU), and computes per-output
cosine. Gate 0.95. This is the board-acceptance metric (real BPU vs FP32).

Usage (host):
  python conversion/scripts/board_compare.py \
      --fp32 conversion/encoder_avatar.onnx \
      --board_out conversion/board_out/encoder \
      --calib conversion/calib_data/encoder_avatar --n 20 --gate 0.95

  # multi-input (net_decode):
  python conversion/scripts/board_compare.py \
      --fp32 conversion/net_decode.onnx \
      --board_out conversion/board_out/net_decode \
      --calib conversion/calib_data/facegen_decode --multi --n 20
"""
import os, sys, glob, argparse
import numpy as np


def cosine(a, b):
    a = a.reshape(-1).astype(np.float64)
    b = b.reshape(-1).astype(np.float64)
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float(np.dot(a, b) / denom)


def load_calib(calib_dir, input_names, multi, n):
    if not multi:
        files = sorted(glob.glob(f"{calib_dir}/*.npy"))[:n]
        if not files:
            raise SystemExit(f"no calib .npy in {calib_dir}")
        name = input_names[0]
        return [{name: np.load(f).astype(np.float32)} for f in files]
    sub = {nm: sorted(glob.glob(f"{calib_dir}/{nm}/*.npy")) for nm in input_names}
    k = min(len(v) for v in sub.values())
    if k == 0:
        raise SystemExit(f"multi calib incomplete: { {n2: len(v) for n2, v in sub.items()} }")
    k = min(k, n)
    return [{nm: np.load(sub[nm][i]).astype(np.float32) for nm in input_names}
            for i in range(k)]


def board_files_for_index(board_out, idx):
    """Return ordered dict {out_name: npy_path} for a given sample index."""
    files = glob.glob(f"{board_out}/{idx:03d}_*.npy")
    out = {}
    for f in files:
        oname = os.path.basename(f)[4:-4]  # strip "iii_" and ".npy"
        out[oname] = f
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fp32", required=True)
    p.add_argument("--board_out", required=True)
    p.add_argument("--calib", required=True)
    p.add_argument("--multi", action="store_true")
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--gate", type=float, default=0.95)
    args = p.parse_args()

    import onnxruntime as ort
    fp = ort.InferenceSession(args.fp32, providers=["CPUExecutionProvider"])
    fp_in = [i.name for i in fp.get_inputs()]
    fp_out = [o.name for o in fp.get_outputs()]
    print(f"FP32 ins={fp_in} outs={fp_out}")

    samples = load_calib(args.calib, fp_in, args.multi, args.n)
    print(f"evaluating {len(samples)} samples from {args.board_out}")

    per_out = [[] for _ in fp_out]
    used = 0
    for i, s in enumerate(samples):
        bfiles = board_files_for_index(args.board_out, i)
        if not bfiles:
            break
        used += 1
        bnames = list(bfiles.keys())
        yf = fp.run(fp_out, {nm: s[nm] for nm in fp_in})
        for j, foname in enumerate(fp_out):
            # match by name, else by order
            key = foname if foname in bfiles else bnames[j]
            barr = np.load(bfiles[key]).astype(np.float32)
            per_out[j].append(cosine(yf[j], barr))

    if used == 0:
        sys.exit(f"no board output files found in {args.board_out}")

    print(f"\n=== board cosine (FP32 ONNX vs HBM on S100 BPU) ===")
    means = []
    for j, foname in enumerate(fp_out):
        arr = np.array(per_out[j])
        label = foname if foname in bfiles else f"{foname}/{bnames[j]}"
        print(f"  [{j}] {label}: mean={arr.mean():.5f} min={arr.min():.5f} (n={len(arr)})")
        means.append(arr.mean())

    overall = float(np.mean(means)) if means else 0.0
    print(f"\nOVERALL board cosine = {overall:.5f}  (gate {args.gate})")
    if overall >= args.gate:
        print("RESULT: PASS — board acceptance OK")
        sys.exit(0)
    else:
        print("RESULT: FAIL — board diverges from FP32")
        sys.exit(1)


if __name__ == "__main__":
    main()
