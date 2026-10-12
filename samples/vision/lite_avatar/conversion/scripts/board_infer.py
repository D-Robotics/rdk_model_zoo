"""
Board-side HBM inference for validation. Runs ON the RDK board (S100 / S100P /
S600) using hbm_runtime. Loads one HBM, feeds calibration .npy inputs, runs
inference on the real BPU, dequantizes outputs, and dumps them to .npy files
for host-side cosine comparison.

Prereq (on board): the hbm_runtime SDK and numpy as shipped in the RDK board
image (no extra installation).
Run on board, e.g. (--n must not exceed the available arrays — the prepared
reference datasets carry audio_encoder 3, audio2mouth 17, facegen_encode 30,
facegen_decoder 30 in the 5-subdir multi layout; fewer files than requested
abort instead of silently running a subset):
  python3 board_infer.py --hbm /path/on/board/encoder_avatar.hbm \
      --calib /path/on/board/calib_data/encoder_avatar --n 3 \
      --out /path/on/board/board_out/encoder
  # multi-input (net_decode):
  python3 board_infer.py --hbm /path/on/board/net_decode.hbm \
      --calib /path/on/board/calib_data/facegen_decode --multi --n 30 \
      --out /path/on/board/board_out/net_decode
"""
import os, sys, glob, argparse
import numpy as np

HB_DTYPE_MAP = {
    "U8": np.uint8, "S8": np.int8, "F32": np.float32, "F16": np.float16,
    "U16": np.uint16, "S16": np.int16, "U32": np.uint32, "S32": np.int32,
    "BOOL8": np.bool_,
}


def load_calib(calib_dir, input_names, multi, n):
    if not multi:
        files = sorted(glob.glob(f"{calib_dir}/*.npy"))
        if not files:
            raise SystemExit(f"no calib .npy in {calib_dir}")
        if len(files) < n:
            raise SystemExit(
                f"calib has {len(files)} sample(s) < requested n={n}"
            )
        name = input_names[0]
        return [{name: np.load(f).astype(np.float32)} for f in files[:n]]
    sub = {nm: sorted(glob.glob(f"{calib_dir}/{nm}/*.npy")) for nm in input_names}
    counts = {n2: len(v) for n2, v in sub.items()}
    if min(counts.values()) == 0:
        raise SystemExit(f"multi calib incomplete: {counts}")
    k = min(counts.values())
    if k < n:
        raise SystemExit(f"multi calib has {k} complete sample(s) < requested n={n}: {counts}")
    return [{nm: np.load(sub[nm][i]).astype(np.float32) for nm in input_names}
            for i in range(n)]


def collect_outputs(out, out_names):
    """Fetch every expected named output; a missing name aborts — never
    substitute a positional/first returned value for a named output."""
    missing = [oname for oname in out_names if oname not in out]
    if missing:
        raise SystemExit(
            f"SDK run returned keys {list(out.keys())}; expected output "
            f"name(s) missing: {missing} — refusing to substitute a "
            "positional value for a named output"
        )
    return {oname: out[oname] for oname in out_names}


def dequant(arr, qinfo):
    """Dequantize a board output to float32 using QuantParams (scale/zero_point/axis)."""
    if qinfo is None or qinfo.quant_type.name != "SCALE":
        return arr.astype(np.float32)
    scale = np.asarray(qinfo.scale, dtype=np.float32)
    zp = np.asarray(qinfo.zero_point, dtype=np.float32)
    axis = qinfo.axis
    arr = arr.astype(np.float32)
    # broadcast scale/zp along axis
    if scale.size == 1:  # per-tensor
        return (arr - zp) * scale
    # per-channel: reshape to broadcast along `axis`
    shape = [1] * arr.ndim
    shape[axis] = arr.shape[axis]
    scale = scale.reshape(shape)
    zp = zp.reshape(shape)
    return (arr - zp) * scale


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hbm", required=True)
    p.add_argument("--calib", required=True)
    p.add_argument("--multi", action="store_true")
    p.add_argument("--n", type=int, default=None,
                   help="sample count; every requested sample must exist "
                        "(n <= available; reference datasets: audio_encoder 3, "
                        "audio2mouth 17, facegen_encode 30, facegen_decoder 30); "
                        "required — no silent default")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    if args.n is None:
        raise SystemExit(
            "--n is required: state the count explicitly (it must not "
            "exceed the available samples; reference datasets: "
            "audio_encoder 3, audio2mouth 17, facegen_encode 30, "
            "facegen_decoder 30)"
        )
    if args.n <= 0:
        raise SystemExit(
            f"--n must be a positive sample count, got {args.n}"
        )

    from hbm_runtime import HB_HBMRuntime
    model = HB_HBMRuntime(args.hbm)
    mname = model.model_names[0]
    ins = model.input_names[mname]
    in_dtypes = model.input_dtypes[mname]
    out_names = model.output_names[mname]
    out_dtypes = model.output_dtypes[mname]
    out_quants = model.output_quants[mname]
    print(f"model={mname}")
    print(f"  inputs={ins}")
    print("  input_dtypes=", {n: in_dtypes[n].name for n in ins})
    print(f"  outputs={out_names}")
    print("  output_dtypes=", {n: out_dtypes[n].name for n in out_names})

    samples = load_calib(args.calib, ins, args.multi, args.n)
    print(f"running {len(samples)} samples on BPU ...")
    os.makedirs(args.out, exist_ok=True)

    for i, s in enumerate(samples):
        # cast inputs to the dtype the model expects (featuremap inputs are F32)
        feed = {}
        for nm in ins:
            dt = HB_DTYPE_MAP.get(in_dtypes[nm].name, np.float32)
            feed[nm] = np.ascontiguousarray(s[nm].astype(dt))
        if len(ins) == 1:
            res = model.run(feed[ins[0]])
        else:
            res = model.run(feed)
        res_all = res[mname]
        if i == 0:
            print("  returned_output_keys=", list(res_all.keys()))
        out = collect_outputs(res_all, out_names)
        for oname in out_names:
            arr = np.asarray(out[oname])
            f = dequant(arr, out_quants.get(oname))
            np.save(os.path.join(args.out, f"{i:03d}_{oname}.npy"), f.astype(np.float32))
        if i == 0:
            for oname in out_names:
                a = np.load(os.path.join(args.out, f"000_{oname}.npy"))
                print(f"  sample0 {oname}: shape={a.shape} dtype={a.dtype} "
                      f"finite={np.isfinite(a).all()} range=[{a.min():.4f},{a.max():.4f}]")

    print(f"done -> {args.out}")


if __name__ == "__main__":
    main()
