# B10 Paraformer application pipeline — host implementation

Status: partial sample implementation. SDK binding, real frontend, native C++,
conversion/evaluation integration and complete sample documentation remain open.
No board, SDK or OE execution, dataset CER or model latency is claimed.

## Composition and preserved behavior

The application pipeline now explicitly composes encoder → predictor → CPU CIF →
decoder. Three injected raw model callables each consume/return named tensors;
`TensorNames` supplies exact physical names, without positional/name guessing.
Validation checks the published fixed batch-one feature/context/predictor/logit
shapes and float32 dtype, before any dependent CPU arithmetic. Copies protect
retained context from runner-buffer reuse. Scheduling and physical-model binding
remain adapter responsibilities, not silently ignored pipeline options.

The decoder uses the source's valid token prefix, greedy argmax, special-token
filter, `@@` removal and concatenation; it does **not** collapse repeated tokens
like CTC. The zero-token CIF fix now has an application outcome: skip the decoder,
return empty text/IDs, mark `decoder_executed=False`, and record decoder timing as
`None`. This is an explicit repaired edge case; the source originally crashed.
Timing separates runner calls and CPU CIF and excludes frontend, binding/loading,
validation/copies outside calls, text decoding and I/O. It is not end-to-end time.

`pipeline.py` is orchestration; `decoding.py` is pure text processing; `cif.py`
remains the shared numerical helper. No artificial all-in-one model `forward`
performs CPU CIF between SDK calls. The published names' INT16 designation is not
used to infer physical I/O dtype. Runtime metadata validation is still required.

## Evidence

Six new behavior tests failed before implementation (retained red log), then pass
together with the seven CIF tests. Tests exercise real orchestration and numerical
helpers with synthetic model boundaries: named feeds/order, padding masking,
repeat/BPE/special-token rules, invalid inputs, zero-token bypass and missing output.
These callables are explicitly **not** model inference results.

The active-manifest vocabulary URL was retrieved with curl after Python urllib's
TLS connection failed; certificate verification was not disabled. Its 93,676 bytes
contain 8,404 unique tokens. The downloaded bytes are preserved as an evidence
fixture; its observed SHA-256 is
`2b20c2b12572d682afff84ce1c8d560f67b8b32a4c1f21567411d141ed352127`.
The publisher manifest remains unchanged, with no published digest assumed.

Twenty randomized logit/count cases produce exactly the same text as the original
S `Paraformer.post_process` method. Verification compiles that exact method from
byte-checked pinned source without importing board dependencies, rather than
rewriting it as a second expected implementation. Both README numerical examples,
both synthetic pipeline examples and both documented test commands execute; all
local README links resolve. See [machine-readable evidence](evidence/2026-09-28-b10-paraformer-pipeline/summary.json)
and [test log](evidence/2026-09-28-b10-paraformer-pipeline/green.log).

Reproduce from repository root using Python with NumPy:

```bash
python docs/releases/unified-migration/evidence/2026-09-28-b10-paraformer-pipeline/verify.py
```

## Remaining scope

Connect exact active-manifest selections, lazy SDK adapters and metadata binding;
actually delegate scheduling to each model; migrate and verify real FunASR frontend
without changing the caller's random state; preserve audio/manifest preparation and
native C++ capabilities; migrate all conversion scripts with shared CIF; add
complete model/root/conversion/evaluator/test-data bilingual instructions. H0–H9
and independent whole-branch review remain open. No batch closure is asserted.
