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
