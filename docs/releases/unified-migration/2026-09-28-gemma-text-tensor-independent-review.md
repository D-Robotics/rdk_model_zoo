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
