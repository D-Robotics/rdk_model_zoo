# B10 Paraformer CPU CIF migration — partial implementation

Status: host implementation of the numerical bridge only; complete sample and
independent review remain open. Board, SDK, OE, frontend equivalence, dataset CER
and latency: **not-run**.

## Source and defect

The S source is pinned to `380e1a2bf42041af54be6f34935e50197cfadff9`.
`platforms/s/samples/speech/paraformer/conversion/cif_numpy.py` is byte-compared
with that commit by the evidence script; the archive is unchanged. With all-zero
weights the source indexes an empty `shift_frames` array and raises `IndexError`.
Total weight below one has the same empty-fire shape. Returning fixed-size zeros
and a zero token count is the meaningful numerical result; whether to bypass the
decoder belongs to the forthcoming pipeline orchestrator.

## Change and architectural boundary

`samples/speech/paraformer/runtime/python/cif.py` provides one pure CPU helper,
with no SDK, file I/O or text decoding. Integration retains the source arithmetic,
float64 accumulation followed by float32 rounding, first-100 truncation and
one-fire-per-frame rule. It explicitly supports the published batch-one shapes;
it does not pretend to fix the source's unsupported batched truncation semantics.
The API requires an explicit `real_T` keyword. Valid inference lengths are 0–400;
`None` is an explicit unmasked-calibration choice. Shape/type/finite/nonnegative
checks reject malformed tensors before arithmetic; inputs are not mutated.

This does not yet wire the helper into either a complete runtime or conversion
pipeline. The remaining migration must make both consumers share it and retain
the Python frontend → C++ inference path. The task forward method must not hide
CPU CIF between multiple model executions: encoder, predictor and decoder need
separate model calls with an explicit CPU bridge in the application pipeline.
No new S600, S100P or X5 support is inferred from the S100 source.

## Verification

The missing implementation first failed all seven behavior tests; original
[red output](evidence/2026-09-28-b10-paraformer-cif/red.log) is retained.
The implemented helper passes seven tests covering independently hand-derived
fractional integration, no-fire input, padding exclusion, explicit calibration,
truncation, ownership, invalid inputs, and 24 byte-exact source comparisons.
The original source no-fire error is independently reproduced and recorded.
Both README examples and both documented unittest commands are executed, their
expected output checked, and all local README links resolved. These are host
numerical checks, not speech recognition.

Reproduce from the repository root with a Python environment containing NumPy:

```bash
python docs/releases/unified-migration/evidence/2026-09-28-b10-paraformer-cif/verify.py
```

[Machine-readable results](evidence/2026-09-28-b10-paraformer-cif/summary.json),
[test output](evidence/2026-09-28-b10-paraformer-cif/green.log),
[English API guide](../../../samples/speech/paraformer/runtime/python/README.md),
[Chinese API guide](../../../samples/speech/paraformer/runtime/python/README_cn.md).

## Still required

Real FunASR frontend parity and isolated RNG handling; three-model binding and
scheduling delegation; vocabulary/manifest validation; strict audio/feature
preparation that never overwrites the user's input manifest; all conversion
scripts/configurations; offline evaluator and historical-result attribution;
source C++ capabilities with resource ownership; complete bilingual root/model/
runtime/conversion/evaluator/test-data instructions. This report closes none of
B10 or H0–H9. Whole-branch independent acceptance remains pending.
