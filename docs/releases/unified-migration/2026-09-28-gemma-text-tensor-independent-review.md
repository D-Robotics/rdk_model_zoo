# Gemma Text tensor contract — independent review in progress

Reviewer: Codex. Base eac59cd6. Status: changes-required. Candidate authored
by Claude Code + GLM; reviewer has made no runtime implementation changes.
Full engine/session organization and remaining H7 scope are not accepted by
this bounded review.

## GEMMA-TEXT-R1 — Zero dimension triggers undefined behavior (P2)

Canonicalize rejects negative dimensions but keeps zero dimensions. For a
F32/NONE inputs_embeds descriptor [0,1536] with positive alignedByteSize=6144
and strides [6144,4], ExpectMatrix reaches FlattenedElements, which divides
INT64_MAX by zero before the invalid shape is rejected. A bad descriptor
therefore aborts instead of producing the promised explicit contract error.

[Standalone fixture](evidence/2026-09-28-gemma-text-tensor-independent-review/zero_dimension.cpp)
compiles the production helper with the existing SDK type fixtures, ASan/UBSan
and no-recover. [Full evidence](evidence/2026-09-28-gemma-text-tensor-independent-review/zero-dimension.json)
records build rc=0, run rc=-6, UBSan division-by-zero at line 120 and the exact
helper SHA-256. No model or vendor SDK is involved. Reject nonpositive
dimensions before shape arithmetic and add meaningful helper/constructor
regression coverage. Assigned back to the original terminal Claude session.

## Review scope so far

Read the helper/header, ModelIo capacity tracking, tensor-copy implementation
and engine binding/append diff. The new descriptor/IO separation and retained
source mask/logit semantics are directionally appropriate, but passing author
suites do not cover the demonstrated zero-dimension case. Remaining ownership,
layout and engine integration review continues after remediation; no broad
acceptance or board/quantization claim is made.

## GEMMA-TEXT-R2 — Adoption leaks on bookkeeping allocation failure (P2)

ModelIo::AddInput and AddOutput receive an already allocated tensor, push its
capacity into a new vector, then put the tensor into the owning vector. If the
capacity push throws std::bad_alloc, the raw tensor is not adopted and Clear
cannot release it. InitModelIo reserves the input/output vectors but not these
new capacity vectors, so the production construction path has this failure gap.

[Independent driver](evidence/2026-09-28-gemma-text-tensor-independent-review/adoption_failure.cpp)
reserves the owning vector as production does and fails the next host allocation.
Both input and output paths catch the exception but release zero buffers on
destruction (expected one). [Output and exact header hash](evidence/2026-09-28-gemma-text-tensor-independent-review/adoption-failure.json)
record build rc=0 and driver rc=1. The driver explicitly frees leaked fixture
buffers after observing the missing owner cleanup.

Make adoption exception-safe with transactional bookkeeping and single ownership
throughout failure. Cover input/output failure positions, subsequent adoption,
normal cleanup and borrowed-KV behavior. Assigned sequentially to the same
Claude session after R1; no competing writer was started on Gemma.

## R1/R2 independent correction check — 2026-09-28

The two **specific findings are corrected** in the current candidate, as recorded in `evidence/2026-09-28-gemma-text-tensor-independent-review/r1-r2-independent-recheck.json`; this is not acceptance of the complete tensor package or H7. Reviewer made no implementation changes.

- R1: recompiling the original zero-dimension reproducer with ASan/UBSan now returns 0 with an explicit nonpositive-dimension rejection, no sanitizer diagnostic. Inspection confirms rejection before flattening arithmetic and a second defensive nonpositive check before division.
- R2: the original allocation-failure reproducer now returns 0 and records exactly one release on both input and output paths. Inspection confirms a temporary RAII owner, capacity rollback after tensor-vector failure and transfer only after both pushes. The permanent adoption test also checks second-allocation failure, subsequent successful adoption/capacity lookup, and repeated clear; existing ownership tests retain borrowed-buffer behavior.
- Fresh whole Gemma host suite: **29 tests pass**. This includes compiled production-source contract/flow and ownership tests against SDK fixtures. No real model, SDK or board was used.

The entire product candidate remains uncommitted pending the remaining tensor/package review; full TextEngine session-stage reorganization and final H7 acceptance are still open. The prior reproduced failures and their original hashes are retained above and in the original evidence.

## Completed bounded tensor-code review — 2026-09-28

The inspected tensor-code package is accepted within host scope after R1/R2 correction. Fresh `tensor-package-native.json` records successful native configure/build and **15 ASan/UBSan CTests**, plus a zero-violation sample checker (no skips/exemptions); the separate R1/R2 record holds **29 whole-sample host tests** and both original counterexamples. The sanitizer build uses the existing local OpenCV and SDK type/call fixtures, not a board SDK.

Reviewed fixed-role type/rank/shape checks, overflow-bounded byte spans, singleton collapse, original allocation capacities, dense KV inputs, padded KV output rows, equal K/V strides, strided writes and greedy logits reads. InitModelIo validates the 35/31 descriptor groups before allocation, makes ownership transactional and pins prefill/decode geometry. Production fill/read paths use the helper; cache alias rejection and per-layer append validation retain borrowed-cache ownership. Source mask quantization and first-maximum logits selection remain explicit algorithms, not extra forward behavior. Matching actual HBM descriptors is not inferred from SDK doubles.

Customer runtime guides describe the new physical tensor boundary and failure behavior accurately, except the nonblocking stale count paragraph (14 CTests / 2 ownership tests, now 15 / 3). A narrowly scoped Claude Code + GLM correction is running before product commit. This bounded tensor acceptance does not accept the still-monolithic session/generation orchestration: masks/preparation, session bookkeeping, debug console output and benchmark composition remain mixed in TextEngine. Those are the next substantive implementation task; H7 remains open.

Final document check: Claude correction is terminal; both runtime guides now accurately state 15 CTests and 3 ownership tests, including allocation-failure adoption. `final-document-check.json` verifies all code/test hashes unchanged since the independent native run. **Bounded tensor package accepted for commit**; session-stage refactoring/H7 remain open.
