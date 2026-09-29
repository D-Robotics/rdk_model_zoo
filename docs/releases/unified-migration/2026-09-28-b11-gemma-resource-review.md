# B11 Gemma SDK resource failure cleanup

Base `c8b0277c`; author implementation record. No board or quantization execution.

## Findings and changes

Host SDK doubles reproduced leaked memory after a failing allocation that returned a pointer,
a leaked task after wait failure, and a leaked packed model after Vision constructor failure.
These three failures are also captured against immutable pre-fix Git source files in the evidence.

`MakeTensor` now rejects nonpositive allocation size/null success and frees memory acquired by
an allocator that returns an error. Vision construction catches failures while members remain
alive and releases partial tensors and the model; normal destruction uses the same cleanup.
Null model handles and unexpected input/output counts are rejected.

Full and selective inference share one task owner and the same dispatch/submit/wait/output
refresh flow. Ownership begins before inference, so an error returning an acquired task also
cleans it up. A normal-path release error propagates without retrying an already released
handle. Destruction during another exception attempts cleanup without masking that exception.

The S600 compiled-core backend, source optional V3 dispatch, and selective input/output index
semantics are preserved. Text/KV callers benefit from the shared task cleanup but their own
constructor/session state has not thereby been independently accepted.

## Evidence

- Four new Python-driven native test groups compile S100 and S600 branches and run 72 host
  cases: 35 S100, 37 S600. S100 does not call the compiled-core query, matching source behavior.
- The same 72 cases pass ASan/UBSan; fixture sets assert no remaining acquired buffers, tasks
  or models and detect duplicate frees. Production/fixture file SHA-256 values are recorded.
- Gemma unittest discovery passes 11 tests, including the previous seven launcher checks.
- Migration contract: 50 samples, 0 violations, 51 policy skips, 0 exemptions.
- README failure semantics and test command are bilingual. No board SDK ABI or real BPU
  execution is inferred from link-time doubles. Optional V3 dynamic-symbol execution itself
  is not exercised by these fixtures.

[Baseline failure reproduction](evidence/2026-09-28-b11-gemma-resources/baseline-reproductions.json),
[sanitizer cases and source bindings](evidence/2026-09-28-b11-gemma-resources/sanitizer-checks.json),
[migration contract](evidence/2026-09-28-b11-gemma-resources/migration-contract.json).

## Remaining scope

Vision still needs strict type/shape/byte-stride/allocation validation, padded output extraction
and removal of the source unknown-type float fallback. Text/KV model ownership and stage
responsibilities, explicit artifact preparation, MiniCPM and the full H0–H9 acceptance remain
open. This closes the reproduced resource failure paths, not the whole native runtime review.
