# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""
Compiler-side cosine eval: FP32 ONNX vs hb_compile's quantized *_ptq_model.onnx.

Why ptq_model.onnx (not the HBM via horizon_tc_ui x86 sim)?
  - hb_compile emits <prefix>_ptq_model.onnx: the INT8/INT16 quantized model as ONNX
    with horizon calibration ops (HzCalibration/HzQuantize/HzDequantize). Plain
    onnxruntime cannot load these graphs ("horizon:HzCalibration(-1) is not a
    registered function/op"); the PTQ session is therefore created exactly the way
    the toolchain's own executor does (hmct.executor.ort2, OE 3.7.0): the shipped
    hmct/executor/horizon_operators.so and onnxruntime_extensions libraries are
    registered via SessionOptions.register_custom_ops_library and the PTQ file is
    fed to onnxruntime unmodified — no calibration op is stripped, replaced or
    renamed, so the comparison keeps true PTQ semantics. No BPU simulator
    daemon/license is needed (the x86 HB_HBMRuntime sim path hangs in the OE
    environment); the FP32 reference stays a plain onnxruntime session.

Multi-input aware: if the model has >1 input, calib must be in the multi-input subdir
layout (calib_dir/<input_name>/k.npy, matched index k). Single-input models use
calib_dir/*.npy flat.

Strictness contract (a comparison either runs honestly or fails loudly):
  * names    — the PTQ graph must expose exactly the FP32 graph's input and
               output name sets; missing or unexpected names abort instead of
               being paired by position (naming/auxiliary differences must be
               investigated, not guessed);
  * samples  — all ``--n`` requested samples must exist; a partial calib set
               never becomes a silent average over fewer samples;
  * shapes   — each output's PTQ shape must equal the FP32 shape;
  * finiteness — non-finite values in either array abort with the offending
               output and sample;
  * roles    — the numerical gate applies to the PRIMARY outputs the runtime
               consumes, declared up front (``--stage`` from this sample's
               four-stage contracts, or ``--primary`` explicitly and
               validated against the graph).  Every output is measured and
               reported: the audio2mouth graph's auxiliary ``feat`` (never
               read by the renderer's postprocess) carries full per-sample
               statistics marked ``not gated`` instead of failing the
               compile check.  Name, shape and finiteness strictness apply
               to ALL outputs, auxiliary included;
  * gate     — PASS requires every PRIMARY output of every compared sample
               to reach the gate (per-sample, per-output minimum; means are
               reported for context, never averaged across outputs).

``--dump-dir`` writes the raw per-output cosine arrays (``<out>.cosine.npy``)
and a ``metrics.json`` summary (per-output role, n/mean/min/max, gate,
verdict), so a receipt can carry measured numbers instead of a console
transcript alone.

Exit codes: 0 pass; 1 comparison ran but the gate was not met (the ONLY
path to 1 — any inability to run, including a missing SDK runtime or an
engine error, exits 2 with the raw error preserved); 2 the comparison
could not run honestly (name/shape/finiteness violations, undeclared or
invalid output roles, missing samples, unreadable models, missing
hmct/onnxruntime_extensions, PTQ session/inference failure).

``--n`` has no default and every requested sample must exist (n must not
exceed the dataset; a smaller explicitly chosen subset is a valid
selection).  The prepared reference datasets for this sample carry
audio_encoder 3, audio2mouth 17, facegen_encode 30 and facegen_decoder 30
(5-subdir multi layout) arrays — the usage examples use those counts;
calibration arrays are generated, not bundled, so a regenerated dataset
may hold a different count.

Usage (in the OE environment, from repo root):
  # single-input (audio2mouth / encoder / net_encode):
  python conversion/scripts/cosine_eval_ptq.py \
      --fp32 conversion/model_1_folded.onnx \
      --ptq  conversion/audio2mouth_output/audio2mouth_ptq_model.onnx \
      --calib conversion/calib_data/audio2mouth \
      --stage audio2mouth --n 17 --gate 0.95 \
      --dump-dir conversion/audio2mouth_output/cosine_dump

  # multi-input (net_decode):
  python conversion/scripts/cosine_eval_ptq.py \
      --fp32 conversion/net_decode.onnx \
      --ptq  conversion/net_decode_output/net_decode_ptq_model.onnx \
      --calib conversion/calib_data/facegen_decode --multi \
      --stage facegen_decoder --n 30 --gate 0.95
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np

#: Primary (runtime-consumed) outputs per stage of this sample's four
#: graphs, keyed by the runtime stage names (avatar.py ``STAGES``).  The
#: audio2mouth folded model also exposes ``feat`` — auxiliary: part of the
#: validated output set, never read by the renderer's postprocess — so it
#: is measured and reported without gating.  Graph outputs beyond a
#: contract's primaries are likewise reported as auxiliary, never silently
#: ignored.
STAGE_PRIMARY_OUTPUTS = {
    "audio_encoder": ("hidden_states",),
    "audio2mouth": ("output",),
    "facegen_encoder": ("f0", "f1", "f2", "f3"),
    "facegen_decoder": ("mouth_image",),
}


class CompareError(Exception):
    """The comparison cannot run honestly; the message says exactly why."""


def resolve_primary_outputs(output_names, stage=None, primary_arg=None):
    """Return the gated output names, explicitly declared and validated.

    ``stage`` selects this sample's known contract; ``primary_arg`` is a
    comma-separated explicit list.  Both given, neither given, an unknown
    stage, or names outside the compared graph abort — the gate is never
    guessed from output order, and no output is silently ignored
    (non-primary outputs stay measured and reported as auxiliary).
    """
    if stage is not None and primary_arg is not None:
        raise CompareError("pass either --stage or --primary, not both")
    if primary_arg is not None:
        names = [part.strip() for part in primary_arg.split(",")]
        if not names or any(not name for name in names):
            raise CompareError("--primary got an empty output name")
        if len(set(names)) != len(names):
            raise CompareError(f"--primary repeats an output: {names}")
        unknown = [name for name in names if name not in output_names]
        if unknown:
            raise CompareError(
                f"--primary names outputs not exposed by the compared graph: "
                f"{unknown} (graph outputs: {list(output_names)})"
            )
        return tuple(names)
    if stage is not None:
        if stage not in STAGE_PRIMARY_OUTPUTS:
            raise CompareError(
                f"unknown stage {stage!r} (known: {', '.join(STAGE_PRIMARY_OUTPUTS)})"
            )
        primaries = STAGE_PRIMARY_OUTPUTS[stage]
        missing = [name for name in primaries if name not in output_names]
        if missing:
            raise CompareError(
                f"stage {stage!r} expects primary outputs {list(primaries)} "
                f"but the graph does not expose {missing} "
                f"(graph outputs: {list(output_names)})"
            )
        return primaries
    raise CompareError(
        "the acceptance gate needs declared primary outputs: pass --stage "
        f"({', '.join(STAGE_PRIMARY_OUTPUTS)}) or --primary NAME[,NAME...]; "
        "every output is measured either way"
    )


def make_fp32_session(path):
    """Plain onnxruntime CPU session for the FP32 reference graph."""
    import onnxruntime as ort

    try:
        return ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    except Exception as exc:
        raise CompareError(f"FP32 session creation failed: {exc!r}") from exc


def make_ptq_session(path):
    """onnxruntime session able to execute OE calibration (PTQ) graphs.

    hb_compile's ``*_ptq_model.onnx`` carries horizon-domain calibration ops
    (HzCalibration/HzQuantize/HzDequantize) that stock onnxruntime refuses to
    load.  The toolchain's own executor — ``hmct.executor.ort2.ORTExecutor2``
    in OE 3.7.0 — runs exactly these graphs by registering its shipped
    custom-op libraries with the stock onnxruntime ``SessionOptions``; this
    mirrors that registration (same libraries, same settings) and feeds the
    PTQ file UNMODIFIED: no calibration op is stripped, replaced by Identity,
    or renamed, so what runs is the real PTQ graph.  Any failure here is a
    cannot-run condition (CompareError keeps the raw error), never a
    measured gate failure.
    """
    import onnxruntime as ort

    try:
        from hmct.executor.ort2 import get_operators_path
        from onnxruntime_extensions import get_library_path
    except Exception as exc:
        raise CompareError(
            "cannot load the SDK custom-op runtime for the PTQ graph "
            f"(hmct/onnxruntime_extensions not importable: {exc!r}); "
            "this script needs the OE toolchain environment"
        ) from exc
    options = ort.SessionOptions()
    # Settings copied from hmct.executor.ort2.create_session: a fixed thread
    # count avoids pthread_setaffinity_np failures inside the toolchain
    # containers; deterministic compute keeps receipts reproducible.
    options.log_severity_level = 3
    options.intra_op_num_threads = 10
    options.use_deterministic_compute = True
    options.register_custom_ops_library(get_operators_path())
    options.register_custom_ops_library(get_library_path())
    try:
        return ort.InferenceSession(
            path, sess_options=options, providers=["CPUExecutionProvider"]
        )
    except Exception as exc:
        raise CompareError(f"PTQ session creation failed: {exc!r}") from exc


def cosine(a, b):
    a = a.reshape(-1).astype(np.float64)
    b = b.reshape(-1).astype(np.float64)
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float(np.dot(a, b) / denom)


def load_calib(calib_dir, input_names, multi, n):
    """Load exactly ``n`` samples in sorted order; never fewer, never guessed."""
    if not multi:
        files = sorted(glob.glob(f"{calib_dir}/*.npy"))
        if not files:
            raise CompareError(f"no calib .npy in {calib_dir}")
        if len(files) < n:
            raise CompareError(
                f"calib has {len(files)} sample(s) < requested n={n}"
            )
        name = input_names[0]
        return [{name: np.load(f).astype(np.float32)} for f in files[:n]]
    sub = {nm: sorted(glob.glob(f"{calib_dir}/{nm}/*.npy")) for nm in input_names}
    counts = {n2: len(v) for n2, v in sub.items()}
    if min(counts.values()) == 0:
        raise CompareError(f"multi calib incomplete: {counts}")
    k = min(counts.values())
    if k < n:
        raise CompareError(f"multi calib has {k} complete sample(s) < requested n={n}: {counts}")
    return [{nm: np.load(sub[nm][i]).astype(np.float32) for nm in input_names}
            for i in range(n)]


def check_name_sets(fp32_names, ptq_names, kind):
    """Require the two graphs to expose the same name set (order may differ)."""
    fp_set, ptq_set = set(fp32_names), set(ptq_names)
    if fp_set == ptq_set:
        return
    missing = sorted(fp_set - ptq_set)
    extra = sorted(ptq_set - fp_set)
    raise CompareError(
        f"{kind} name sets differ between FP32 and PTQ graphs "
        f"(cannot pair by position): missing-in-ptq={missing} "
        f"unexpected-in-ptq={extra} (fp32={list(fp32_names)} ptq={list(ptq_names)})"
    )


def evaluate(run_fp32, run_ptq, input_names, output_names, samples):
    """Compare every sample's every output; return {out_name: [cosine]}.

    ``run_fp32``/``run_ptq`` map a feed dict to their graph's outputs in
    ``output_names`` order (onnxruntime in production, doubles in tests).
    """
    per_out = {name: [] for name in output_names}
    for i, sample in enumerate(samples):
        feed = {nm: sample[nm] for nm in input_names}
        fp32 = dict(zip(output_names, run_fp32(feed)))
        ptq = dict(zip(output_names, run_ptq(feed)))
        for name in output_names:
            fp = np.asarray(fp32[name], dtype=np.float32)
            pq = np.asarray(ptq[name], dtype=np.float32)
            if fp.shape != pq.shape:
                raise CompareError(
                    f"output {name!r} sample {i:03d}: shape mismatch "
                    f"fp32={fp.shape} ptq={pq.shape}"
                )
            if not np.all(np.isfinite(fp)):
                raise CompareError(
                    f"FP32 output {name!r} sample {i:03d} contains non-finite values"
                )
            if not np.all(np.isfinite(pq)):
                raise CompareError(
                    f"PTQ output {name!r} sample {i:03d} contains non-finite values"
                )
            per_out[name].append(cosine(fp, pq))
    return per_out


def report(per_out, gate, primary_outputs):
    """Print per-output measured results; return True iff the gate is met.

    Only ``primary_outputs`` are gated; every other output is printed with
    its full statistics as ``auxiliary: measured, not gated``.
    """
    print(f"\n=== cosine (FP32 vs PTQ quantized ONNX) ===")
    passed = True
    for name, values in per_out.items():
        arr = np.asarray(values, dtype=np.float64)
        if name in primary_outputs:
            ok = arr.min() >= gate
            passed = passed and ok
            verdict = "ok" if ok else "BELOW GATE"
        else:
            verdict = "auxiliary: measured, not gated"
        print(
            f"  {name}: n={len(arr)} mean={arr.mean():.5f} "
            f"min={arr.min():.5f} max={arr.max():.5f} [{verdict}]"
        )
    print(
        f"\ngate {gate} applies per sample and per PRIMARY output "
        f"(primaries: {', '.join(primary_outputs)}; no averaging; "
        "auxiliary outputs are measured and reported, not gated)"
    )
    return passed


def dump(per_out, gate, dump_dir, fp32_path, ptq_path, primary_outputs):
    """Write raw per-output cosine arrays plus a metrics.json summary."""
    os.makedirs(dump_dir, exist_ok=True)
    outputs = {}
    for name, values in per_out.items():
        arr = np.asarray(values, dtype=np.float64)
        safe = re.sub(r"[^A-Za-z0-9._-]", "_", name)
        np.save(os.path.join(dump_dir, f"{safe}.cosine.npy"), arr)
        outputs[name] = {
            "role": "primary" if name in primary_outputs else "auxiliary",
            "n": int(arr.size),
            "mean": float(arr.mean()),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "gate": gate if name in primary_outputs else None,
            "pass": bool(arr.min() >= gate) if name in primary_outputs else None,
        }
    metrics = {
        "comparison": "fp32_onnx_vs_ptq_model_onnx",
        "fp32": fp32_path,
        "ptq": ptq_path,
        "n_samples": max(v["n"] for v in outputs.values()),
        "gate": gate,
        "gate_semantics": (
            "per sample and per PRIMARY output (minimum, no averaging); "
            "auxiliary outputs are measured and reported, not gated"
        ),
        "primary_outputs": list(primary_outputs),
        "outputs": outputs,
        "passed": all(v["pass"] for v in outputs.values() if v["role"] == "primary"),
    }
    with open(os.path.join(dump_dir, "metrics.json"), "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print(f"metrics and raw cosine arrays written under {dump_dir}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fp32", required=True)
    p.add_argument("--ptq", required=True)
    p.add_argument("--calib", required=True)
    p.add_argument("--multi", action="store_true", help="multi-input subdir calib layout")
    p.add_argument(
        "--n",
        type=int,
        default=None,
        help="sample count; every requested sample must exist (n <= available; "
        "reference datasets: audio_encoder 3, audio2mouth 17, facegen_encode 30, "
        "facegen_decoder 30); required — no silent default",
    )
    p.add_argument("--gate", type=float, default=0.95)
    p.add_argument("--dump-dir", default=None,
                   help="write per-output cosine .npy arrays and metrics.json here")
    p.add_argument(
        "--stage",
        choices=sorted(STAGE_PRIMARY_OUTPUTS),
        help="declare the primary outputs via this sample's stage contract",
    )
    p.add_argument(
        "--primary",
        default=None,
        help="declare the gated outputs explicitly (comma-separated)",
    )
    args = p.parse_args()

    try:
        if args.n is None:
            raise CompareError(
                "--n is required: state the count explicitly (it must not "
                "exceed the available samples; reference datasets: "
                "audio_encoder 3, audio2mouth 17, facegen_encode 30, "
                "facegen_decoder 30)"
            )
        if args.n <= 0:
            raise CompareError(
                f"--n must be a positive sample count, got {args.n}"
            )
        fp = make_fp32_session(args.fp32)
        pq = make_ptq_session(args.ptq)
        fp_in = [i.name for i in fp.get_inputs()]
        pq_in = [i.name for i in pq.get_inputs()]
        fp_out = [o.name for o in fp.get_outputs()]
        pq_out = [o.name for o in pq.get_outputs()]
        print(f"FP32 ins={fp_in} outs={fp_out}")
        print(f"PTQ  ins={pq_in} outs={pq_out}")
        check_name_sets(fp_in, pq_in, "input")
        check_name_sets(fp_out, pq_out, "output")
        primary_outputs = resolve_primary_outputs(fp_out, args.stage, args.primary)
        print(f"primary (gated) outputs: {list(primary_outputs)}")
        samples = load_calib(args.calib, pq_in, args.multi, args.n)
        print(f"evaluating on {len(samples)} samples")
        per_out = evaluate(
            lambda feed: fp.run(fp_out, feed),
            # Request the PTQ outputs BY the FP32 graph's names: the two
            # graphs expose the same name set (checked above) but their
            # metadata orders may differ, and onnxruntime returns values in
            # the requested order — outputs pair by name, never position.
            lambda feed: pq.run(fp_out, feed),
            pq_in,
            fp_out,
            samples,
        )
        if args.dump_dir:
            dump(per_out, args.gate, args.dump_dir, args.fp32, args.ptq,
                 primary_outputs)
        passed = report(per_out, args.gate, primary_outputs)
    except CompareError as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(2)
    except Exception as error:  # noqa: BLE001 — ANY inability to run must exit 2
        import traceback

        traceback.print_exc()
        print(f"cannot-run (unhandled, NOT a measured gate failure): {error!r}",
              file=sys.stderr)
        sys.exit(2)

    if passed:
        print("RESULT: PASS")
        sys.exit(0)
    print("RESULT: FAIL — if INT8, switch to INT16 (quant_config.model_config.all_node_type: int16)")
    sys.exit(1)


if __name__ == "__main__":
    main()
