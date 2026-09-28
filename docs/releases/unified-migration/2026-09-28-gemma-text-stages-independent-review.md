# Gemma Text stages — independent review, changes required

Codex inspected the new input/session/transport modules and engine composition.
The separation is substantive: prepared CPU vectors, raw SDK transport and
explicit KV/logit processing are distinct. Fresh independent checks pass all
30 Python host tests and 19 existing ASan/UBSan CTests, with candidate hashes
stable. These tests do not cover the two public API memory errors below.
See `evidence/2026-09-28-gemma-text-stages-independent-review/` for full outputs,
source drivers and sanitizer traces. No board/vendor SDK/weights were used.

## TEXT-R1 — unchecked mask geometry writes before allocation (P1)

PrepareBatchInputs(embeddings, {11}, 4096, 1, nullptr, kChunkSize) reaches
BuildFullMask with negative cache_col_start. ASan reports a 4-byte write 1020
bytes before the 4 MiB allocation at gemma4_text_inputs.cpp:42. This API is now
public and documented; the fixed 4096-token context must fail safely before
buffer access. Validate signed dimensions, valid counts, token count versus
sequence, chunk/decode positions and supported mask geometry before allocation/
lookup/write, with overflow-safe arithmetic. Public mask helpers must have a
clear, safe contract too. Preserve valid existing context/continuation behavior,
including the documented limit and alignment rules, rather than arbitrary
clamping that silently changes attention. Cover boundary values and malformed
prepared inputs with real sanitizers; the original driver must reject cleanly.

## TEXT-R2 — public continuation reads past short hidden vector (P1)

Using the existing host SDK/embedding double, construct TextEngine and call
ContinueGenerate({11,22}, 1, &std::vector<float>(1,0)). ASan reports a 6144-byte
read beyond the four-byte vector through PrepareBatchInputs. The one-shot
GenerateWithPromptEmbeddings checks length, but public ContinueGenerate and
ContinueGenerateStream do not. Validate full hidden extent before any session
mutation or stage read. Make the newly public preparation API able to establish
its hidden-buffer extent (a raw pointer without length cannot validate it).
Preserve optional no-hidden behavior and valid image embedding use; cover both
public continuation entry points, nonzero prefix offsets and session reuse after
rejection. Do not silently synthesize missing rows or accept suffix-only storage
when the implementation indexes the whole prompt.

## TEXT-DOC-R1 — public header misstates return/hidden semantics (P2)

Generate/GenerateStream Doxygen still says output excludes the prompt, but the
implemented and tested behavior returns full prompt + generated tokens.
ContinueGenerate/Stream calls full_hidden a suffix although preparation indexes
chunk_start rows of the entire prompt. Correct public header and paired runtime
guides with the actual full-vector, size/lifetime and rejection contract. Keep
CLI generation and benchmark semantics. The always-true unittest assertion
assertIn("passed", run.stdout + "passed") should also be removed or replaced
with an observable assertion; it cannot verify any output.

This package is not accepted and H7 stays open. Original tensor/ownership fixes
must remain intact. Product remediation belongs to Claude Code + GLM; Codex will
rerun the exact two original drivers and affected checks independently.
