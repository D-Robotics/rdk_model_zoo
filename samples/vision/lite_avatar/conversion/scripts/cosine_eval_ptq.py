"""
Generic cosine-similarity eval: FP32 ONNX vs hb_compile's quantized *_ptq_model.onnx.

Why ptq_model.onnx (not the HBM via horizon_tc_ui x86 sim)?
  - hb_compile emits <prefix>_ptq_model.onnx: the INT8/INT16 quantized model as ONNX
    (QuantizeLinear/DequantizeLinear/QLinearConv nodes). It runs on plain onnxruntime
    CPU — no BPU simulator daemon/license needed (the x86 HB_HBMRuntime sim path hangs
    in the OE environment). This is the authoritative per-tensor cosine check the toolchain
    itself uses; comparing FP32 vs ptq quantized ONNX on identical calib inputs gives the
    same cosine hb_compile reports internally.

Multi-input aware: if the model has >1 input, calib must be in the multi-input subdir
layout (calib_dir/<input_name>/k.npy, matched index k). Single-input models use
calib_dir/*.npy flat.

Pass gate: mean cosine across all outputs >= gate (default 0.95).

Usage (in the OE environment, from repo root):
  # single-input (audio2mouth / encoder / net_encode):
  python conversion/scripts/cosine_eval_ptq.py \
      --fp32 conversion/model_1_folded.onnx \
      --ptq  conversion/audio2mouth_output/audio2mouth_ptq_model.onnx \
      --calib conversion/calib_data/audio2mouth --n 17 --gate 0.95

  # multi-input (net_decode):
  python conversion/scripts/cosine_eval_ptq.py \
      --fp32 conversion/net_decode.onnx \
      --ptq  conversion/net_decode_output/net_decode_ptq_model.onnx \
      --calib conversion/calib_data/facegen_decode --multi --n 30 --gate 0.95
"""
import os, sys, glob, argparse
import numpy as np


def cosine(a, b):
    a = a.reshape(-1).astype(np.float64)
    b = b.reshape(-1).astype(np.float64)
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float(np.dot(a, b) / denom)


def load_calib(calib_dir, input_names, multi, n):
    """Return list of dict {input_name: np.ndarray} (one per sample)."""
    if not multi:
        files = sorted(glob.glob(f"{calib_dir}/*.npy"))[:n]
        if not files:
            raise SystemExit(f"no calib .npy in {calib_dir}")
        name = input_names[0]
        return [{name: np.load(f).astype(np.float32)} for f in files]
    # multi-input: one subdir per input, matched index
    sub = {nm: sorted(glob.glob(f"{calib_dir}/{nm}/*.npy")) for nm in input_names}
    k = min(len(v) for v in sub.values())
    if k == 0:
        raise SystemExit(f"multi-input calib incomplete in {calib_dir}: { {n2: len(v) for n2,v in sub.items()} }")
    k = min(k, n)
    samples = []
    for i in range(k):
        d = {nm: np.load(sub[nm][i]).astype(np.float32) for nm in input_names}
        samples.append(d)
    return samples


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fp32", required=True)
    p.add_argument("--ptq", required=True)
    p.add_argument("--calib", required=True)
    p.add_argument("--multi", action="store_true", help="multi-input subdir calib layout")
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--gate", type=float, default=0.95)
    args = p.parse_args()

    import onnxruntime as ort
    fp = ort.InferenceSession(args.fp32, providers=["CPUExecutionProvider"])
    pq = ort.InferenceSession(args.ptq, providers=["CPUExecutionProvider"])

    fp_in = [i.name for i in fp.get_inputs()]
    pq_in = [i.name for i in pq.get_inputs()]
    fp_out = [o.name for o in fp.get_outputs()]
    pq_out = [o.name for o in pq.get_outputs()]
    print(f"FP32 ins={fp_in} outs={fp_out}")
    print(f"PTQ  ins={pq_in} outs={pq_out}")

    # use the PTQ input names as the calib subdir keys (they match the ONNX input_names)
    calib_keys = pq_in
    samples = load_calib(args.calib, calib_keys, args.multi, args.n)
    print(f"evaluating on {len(samples)} samples")

    # pair outputs by index (FP32 and PTQ output orders should match)
    n_out = min(len(fp_out), len(pq_out))
    per_out = [[] for _ in range(n_out)]

    for i, s in enumerate(samples):
        # feed FP32 with its input names; PTQ with its input names. Re-map calib keys.
        fp_feed = {nm: s[k] for nm, k in zip(fp_in, calib_keys)} if fp_in != calib_keys else s
        pq_feed = s
        yf = fp.run(fp_out, fp_feed)
        yp = pq.run(pq_out, pq_feed)
        for j in range(n_out):
            per_out[j].append(cosine(yf[j], yp[j]))

    print(f"\n=== cosine (FP32 vs PTQ quantized ONNX) ===")
    means = []
    for j in range(n_out):
        arr = np.array(per_out[j])
        print(f"  [{j}] {fp_out[j]} / {pq_out[j]}: mean={arr.mean():.5f} min={arr.min():.5f}")
        means.append(arr.mean())

    overall = float(np.mean(means))
    print(f"\nOVERALL mean cosine = {overall:.5f}  (gate {args.gate})")
    if overall >= args.gate:
        print("RESULT: PASS")
        sys.exit(0)
    else:
        print("RESULT: FAIL — if INT8, switch to INT16 (quant_config.model_config.all_node_type: int16)")
        sys.exit(1)


if __name__ == "__main__":
    main()
