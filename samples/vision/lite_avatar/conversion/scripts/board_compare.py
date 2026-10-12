"""
Host-side cosine comparison: FP32 ONNX vs board HBM outputs (dumped by board_infer.py).

Loads the board's dequantized output .npy files (named <idx>_<out_name>.npy),
runs the FP32 ONNX on the same calib inputs (onnxruntime CPU), and computes
per-output cosine similarity on real calibrated inputs. This is the
board-acceptance metric (real BPU vs FP32) for every target the artifacts
were compiled for.

Strictness contract (a comparison either runs honestly or fails loudly):
  * names    — every compared sample must carry exactly the FP32 graph's
               output-name set; missing or unexpected names abort instead of
               being paired by glob order (naming/auxiliary differences must
               be investigated, not guessed);
  * samples  — all ``--n`` requested samples must exist on both sides
               (calib inputs and board dumps); a partial set never becomes a
               silent average;
  * shapes   — each output's board shape must equal the FP32 shape;
  * finiteness — non-finite values in either array abort with the offending
               output and sample;
  * roles    — the numerical gate applies to the PRIMARY outputs the
               runtime consumes, declared up front (``--stage`` from this
               sample's four-stage contracts, or ``--primary`` explicitly
               and validated against the graph).  Every output is measured
               and reported: the audio2mouth graph's auxiliary ``feat``
               (never read by the renderer's postprocess) carries full
               per-sample statistics marked ``not gated`` instead of
               failing delivery.  Name, shape and finiteness strictness
               apply to ALL outputs, auxiliary included;
  * gate     — PASS requires every PRIMARY output of every compared sample
               to reach the gate (per-sample, per-output minimum; means are
               reported for context, never averaged across outputs).

Exit codes: 0 pass; 1 comparison ran but the gate was not met (the ONLY
path to 1 — any inability to run, including an engine error, exits 2 with
the raw error preserved); 2 the comparison could not run honestly (missing
samples/outputs, name/shape/finiteness violations, undeclared or invalid
output roles, unreadable inputs).

Usage (host).  Every requested sample must exist (n must not exceed the
dataset; smaller explicitly chosen subsets are valid); strict loading
aborts instead of truncating.  The prepared reference datasets carry
audio_encoder 3, audio2mouth 17, facegen_encode 30, facegen_decoder 30
(5-subdir multi layout) arrays — these examples use those counts:
  python conversion/scripts/board_compare.py \
      --fp32 conversion/encoder_avatar.onnx \
      --board_out conversion/board_out/encoder \
      --calib conversion/calib_data/encoder_avatar \
      --stage audio_encoder --n 3 --gate 0.95

  # multi-input (net_decode):
  python conversion/scripts/board_compare.py \
      --fp32 conversion/net_decode.onnx \
      --board_out conversion/board_out/net_decode \
      --calib conversion/calib_data/facegen_decode --multi \
      --stage facegen_decoder --n 30

  # audio2mouth: ``output`` is gated; the auxiliary ``feat`` is measured
  # and reported, not gated:
  python conversion/scripts/board_compare.py \
      --fp32 conversion/model_1_folded.onnx \
      --board_out conversion/board_out/audio2mouth \
      --calib conversion/calib_data/audio2mouth \
      --stage audio2mouth --n 17
"""
import argparse
import glob
import os
import re
import sys

import numpy as np

BOARD_FILE_RE = re.compile(r"^(\d+)_(.+)\.npy$")

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


def cosine(a, b):
    a = a.reshape(-1).astype(np.float64)
    b = b.reshape(-1).astype(np.float64)
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float(np.dot(a, b) / denom)


def load_calib(calib_dir, input_names, multi, n):
    """Load the first ``n`` samples in the same sorted order board_infer ran."""
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


def pair_board_samples(board_out, count, output_names):
    """Return [{out_name: npy_path}] for sample indices 0..count-1, strictly.

    Aborts (CompareError) when a requested index is missing, when a sample's
    dumped name set differs from the FP32 output set, or when an index is
    dumped twice. Dumps beyond the requested range are reported, not used.
    """
    if not os.path.isdir(board_out):
        raise CompareError(f"board output directory not found: {board_out}")
    by_index = {}
    for path in sorted(glob.glob(f"{board_out}/*.npy")):
        base = os.path.basename(path)
        m = BOARD_FILE_RE.match(base)
        if m is None:
            raise CompareError(
                f"unrecognized board dump name {base!r} in {board_out} "
                "(expected <idx>_<out_name>.npy from board_infer.py)"
            )
        idx, oname = int(m.group(1)), m.group(2)
        if idx in by_index and oname in by_index[idx]:
            raise CompareError(f"board dump {base!r}: duplicate output for index {idx}")
        by_index.setdefault(idx, {})[oname] = path

    wanted = set(output_names)
    pairs = []
    for i in range(count):
        files = by_index.get(i, {})
        if not files:
            raise CompareError(
                f"board dump index {i:03d} missing in {board_out} "
                f"(requested n={count}; found indices {sorted(by_index)})"
            )
        missing = wanted - set(files)
        extra = set(files) - wanted
        if missing or extra:
            raise CompareError(
                f"board dump index {i:03d} output names differ from the FP32 "
                f"graph: missing={sorted(missing)} unexpected={sorted(extra)} "
                f"(fp32 outputs={list(output_names)})"
            )
        pairs.append(files)
    unused = sorted(set(by_index) - set(range(count)))
    if unused:
        print(
            f"note: board dump carries {len(unused)} index(es) beyond the "
            f"requested n={count} (first={unused[0]:03d}); not compared"
        )
    return pairs


def evaluate(run, input_names, output_names, samples, board_out):
    """Compare every sample's every output; return {out_name: [cosine]}.

    ``run`` maps a feed dict to the FP32 graph's outputs in
    ``output_names`` order (onnxruntime in production, a double in tests).
    """
    pairs = pair_board_samples(board_out, len(samples), output_names)
    per_out = {name: [] for name in output_names}
    for i, (sample, files) in enumerate(zip(samples, pairs)):
        fp32 = dict(zip(output_names, run({nm: sample[nm] for nm in input_names})))
        for name in output_names:
            board = np.load(files[name]).astype(np.float32)
            fp = np.asarray(fp32[name], dtype=np.float32)
            if fp.shape != board.shape:
                raise CompareError(
                    f"output {name!r} sample {i:03d}: shape mismatch "
                    f"fp32={fp.shape} board={board.shape}"
                )
            if not np.all(np.isfinite(fp)):
                raise CompareError(
                    f"FP32 output {name!r} sample {i:03d} contains non-finite values"
                )
            if not np.all(np.isfinite(board)):
                raise CompareError(
                    f"board output {name!r} sample {i:03d} contains non-finite values"
                )
            per_out[name].append(cosine(fp, board))
    return per_out


def report(per_out, gate, primary_outputs):
    """Print per-output measured results; return True iff the gate is met.

    Only ``primary_outputs`` are gated; every other output is printed with
    its full statistics as ``auxiliary: measured, not gated``.
    """
    print(f"\n=== board cosine (FP32 ONNX vs HBM on the BPU) ===")
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fp32", required=True)
    p.add_argument("--board_out", required=True)
    p.add_argument("--calib", required=True)
    p.add_argument("--multi", action="store_true")
    p.add_argument(
        "--n",
        type=int,
        default=None,
        help="sample count; every requested sample must exist (n <= available; "
        "reference datasets: audio_encoder 3, audio2mouth 17, facegen_encode 30, "
        "facegen_decoder 30); required — no silent default",
    )
    p.add_argument("--gate", type=float, default=0.95)
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
        import onnxruntime as ort

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
        session = ort.InferenceSession(args.fp32, providers=["CPUExecutionProvider"])
        input_names = [i.name for i in session.get_inputs()]
        output_names = [o.name for o in session.get_outputs()]
        print(f"FP32 ins={input_names} outs={output_names}")
        primary_outputs = resolve_primary_outputs(
            output_names, args.stage, args.primary
        )
        print(f"primary (gated) outputs: {list(primary_outputs)}")
        samples = load_calib(args.calib, input_names, args.multi, args.n)
        print(f"evaluating {len(samples)} samples from {args.board_out}")
        per_out = evaluate(
            lambda feed: session.run(output_names, feed),
            input_names,
            output_names,
            samples,
            args.board_out,
        )
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
        print("RESULT: PASS — board acceptance OK")
        sys.exit(0)
    print("RESULT: FAIL — board diverges from FP32")
    sys.exit(1)


if __name__ == "__main__":
    main()
