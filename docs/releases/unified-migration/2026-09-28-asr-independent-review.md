# ASR independent host review — runtime scope accepted

Reviewer: Codex. Candidate base `cce19f06`. Existing 21 Python tests and five
ASan/UBSan native CTests pass; this does not cover the reproduced defect below.
[Full commands, output and candidate hashes](evidence/2026-09-28-asr-independent-review/verification.json).
Native scope includes contract, actual host audio frontend, task, SDK fixture and
preflight, not a complete vendor SDK build or new board transcription. No model,
quantization toolchain or dependencies were downloaded/installed.

## ASR-R1 — int32 logits round into a blank-token tie (P2)

The Python binding explicitly accepts int32 SCALE output. postprocess.transcribe
passes it through default float32 affine decoding and forces float32 once more
before decode_logits; that decoder also only permits float32. Two distinct raw
scores 16777216 and 16777217 therefore become identical. With scale 1, offset 0,
shape [1,1,3503], first score at blank ID 0 and second at token ID 1, production
transcribe returns an empty string instead of token1. Actual binding accepts the
metadata. [Independent reproducer](evidence/2026-09-28-asr-independent-review/int32-argmax.json).
This is a synthetic host boundary, not evidence that the published HBM emits this
particular pair. It disproves the currently advertised general integer contract.

Preserve integer affine comparison precision through argmax and then apply the
existing ID decoder. Using the existing float64-capable shared helper is possible,
but merely changing its argument is insufficient while the next layer forces
float32. Keep raw F32 semantics, CTC collapse-before-blank removal, legacy mode,
per-chunk state reset and shared defaults for unrelated samples. Do not remove
integer support to avoid fixing the claimed contract. Add meaningful scalar and
per-channel scale/offset cases, true ties and both decoding modes; update the
paired runtime documentation for comparison precision. Run the bounded ASR suite
and original counterexample; native code is unaffected unless changed.

## Other reviewed observations and limits

The task keeps frontend/transport/decoding separate. Chunk geometry is explicit;
this is independent-window processing, not hidden model streaming state. The
source Python Fourier resampler and native sinc resampler differ deliberately,
and the guide discloses that their outputs are not guaranteed identical. Native
adapter inspection found explicit rank/type/stride/capacity validation, owned
compact copies and scoped SDK resources. Identity/model/vocabulary preflight is
separate from task math. Historical source evidence and failed partial-file
reports are distinguished from new successful model results.

These observations do not close full ASR or B10. ASR-R1 remains open pending
Claude implementation and independent recheck. Trusted conversion/evaluation
recipes are not being rerun; no board, real SDK ABI or dataset precision claim is
made by this host review.

## Independent recheck — ASR-R1 closed (2026-09-28)

After Claude Code + GLM exited, Codex reviewed all six changed ASR files and ran
27 Python tests plus the sample checker (0 violations, 1 CLI policy skip,
0 exemptions). The original [1,1,3503] int32 score pair now returns token1 in both
CTC and legacy modes through actual bind_model and ASR.post_process. Raw tensor
bytes are unchanged; all 62 sample file hashes were stable during this check.
[Commands, outputs, counterexample and hashes](evidence/2026-09-28-asr-independent-review/r1-recheck.json).

The integer branch explicitly requests float64 affine decoding and carries that
dtype through argmax into the existing ID decoder. The original float32 decoder
and ID-decoding behavior remain unchanged. Tests cover binding plus predict,
per-channel scale/offset, true ties, both modes and F32 vestigial descriptors.
Shared quantization defaults and native runtime files were not changed; native
hashes still match the prior independent review, so its five native CTests are
retained without an unrelated rerun. They retain their original limited scope.

The helper name decode_exact_logits refers to the comparison path used for this
integer precision fix; it is not an arbitrary-precision arithmetic guarantee.
Different channel transforms may change raw ranking, and this evidence establishes
the reviewed regression and tested affine cases rather than proving all numerical
quantization configurations. Existing failure evidence remains historical.

Disposition: close ASR-R1 and accept the reviewed ASR host runtime/associated
runtime-documentation scope. B10/H6 and whole-branch acceptance remain separate.
No actual board, compiled model, real SDK ABI, corpus accuracy or quantization
recipe run occurred. Conversion recipes remain trusted source documentation.
