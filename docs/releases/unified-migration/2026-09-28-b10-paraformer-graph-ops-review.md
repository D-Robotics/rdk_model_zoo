# Paraformer shared conversion graph operations

2026-09-28. Author implementation/host verification, not independent acceptance.
Base `8c232c93600ea35a823ff090a8349e85132462e5`.

The conversion directory now provides non-mutating graph dependency sorting,
Gather index adaptation, provable constant Range folding and per-consumer axis
normalization. Graph utilities remain outside model inference. They do not yet
constitute a full exporter, stage extractor, calibration or OE workflow.

The source Gather pass changes a shared INT64 initializer in place, invalidating
an unrelated INT64 Add consumer. The source topological sorter saves a partial
empty graph on a cycle before its checker fails. `verify.py` reproduces both
against the preserved source, records source file hashes and checks the actual
failure. The new implementation preserves shared values and rejects incomplete
graphs before returning a result. No archived source files are changed.

Range folding uses deterministic constant/known-shape proof, not one sampled
execution. Dynamic INT64 Gather casting requires explicit opt-in and an external
range contract; it does not insert runtime checks. Direct constants exceeding the
evaluation limit remain errors even with opt-in. `constant-budget-red.log` records
the failing regression before this last fix. Flat graphs and supported constant
operations are an explicit boundary; no general optimizer or sandbox is claimed.

## Evidence and reproduction

[Evidence directory](evidence/2026-09-28-b10-paraformer-graph-ops/):

- `red.log`: initial graph module absent, tests fail.
- `constant-budget-red.log`: oversized constant incorrectly allowed as dynamic,
  reproduced before repair.
- `tests.log`: 48 sample tests passed, including nine graph tests executing with
  actual ONNX and ONNX Runtime dependencies, no skips. Numerical graph comparisons
  require equal output dtype and array values; no speech weights are involved.
- `verify.py`, `verify.log`, `summary.json`: two source defects reproduced; both
  complete bilingual Python API examples executed and local links checked. The
  actual graph dependency versions are recorded separately from the FunASR stack.
- `migration-gate.json`/`.log`: current migration gate; Paraformer remains pending
  and is not represented as a completed sample by this gate.

From the repository root, with NumPy/ONNX/ONNX Runtime installed:

```bash
python -m unittest discover -s samples/speech/paraformer/tests -v
python docs/releases/unified-migration/evidence/2026-09-28-b10-paraformer-graph-ops/verify.py
python tools/sample_contract/check.py --scope migration --parser-mode import --report /tmp/paraformer-graph-migration-gate.json
```

The conversion README pair provides dependency distinctions, full API contracts,
an executable example, a source-stage migration table, failure behavior and the
remaining workflow. Neither language is only a link to the other. Historical
source recipe settings are retained without claiming fresh CER or HBM evidence.

Full-weight export, safe stage extraction, simplifier checks, calibration,
compiler integration, dedicated evaluator and final conversion documentation
remain pending. Real weights/OE/SDK/board validation is not-run for this work.
Paraformer Refactor/Docs remain pending, independent Review=not-run, Closed=no;
B10 and H0–H9 remain open.
