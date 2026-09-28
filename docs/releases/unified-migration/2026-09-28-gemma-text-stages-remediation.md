# B11 Gemma Text stage separation — author record

Base: `af258e22` (working tree; the tensor-contract package is independently
accepted). Implementation by Claude Code + GLM; Codex reviews and owns Git
synchronization. Board, real SDK ABI, and quantization remain not-run and
out of scope. This record is written in two passes: the design and file
responsibility map below were fixed before implementation; verification
sections are filled from actual runs.

## Problem

`gemma4_text_engine.cpp` still concentrated every Text responsibility in one
translation unit: CPU input preparation (masks, positions, embeddings),
strided SDK tensor writes, inference/task lifecycle, KV output gathering,
greedy decoding, session/policy decisions (context shift, auto-truncate,
continuation alignment), and an implicit `GEMMA4_DEBUG` → `std::cerr` print
inside input preparation. The accepted public API is mature and used by all
five executables; the separation must not change it.

## Design and file responsibility map (recorded before implementation)

### Stage decomposition

1. **Stage 1 — pure CPU input preparation** → `inc/gemma4_text_inputs.hpp`,
   `src/gemma4_text_inputs.cpp`. Produces a plain `TextBatchInputs` value
   (`token_ids` int64[seq], `hidden` float[seq×1536], `positions`
   int32[seq], `full_mask_q`/`slide_mask_q` int16[seq×4096]) via
   `PrepareBatchInputs(...)` / `PrepareDecodeInputs(...)`. Owns the moved
   verbatim mask builders `BuildFullMask` / `BuildSlidingMask` /
   `QuantizeMask` (they never read the `KvCache&` parameter — recorded as a
   source fact, not changed). PLE image-token substitution (249560 → pad 0),
   embedding lookup, and the prebuilt-hidden override for image slots stay
   exactly as the source implemented them. No SDK types, no printing.
2. **Stage 2 — raw SDK inference/transport** →
   `inc/gemma4_text_transport.hpp`, `src/gemma4_text_transport.cpp`.
   `InitTextSubgraph` (descriptor validation + allocation + role mapping,
   moved from the engine), `BindKvCache` (cross-subgraph cache size
   agreement + zero-copy borrowing with capacities), `WriteBatchInputs`
   (strided writes of the five ordinary inputs through the accepted
   `gemma4_text_tensor` helpers), `RunSubgraphInference` (selective flush of
   all 35 inputs, task lifecycle, output refresh), `CollectKvOutputs`
   (post-inference descriptor revalidation against owned capacities + the
   shared K/V row-stride rule). No hidden decoding, no user-facing IO.
3. **Stage 3 — explicit output decoding and KV/session update** → stays with
   the engine as explicit orchestration steps: decoding is the accepted
   `ArgmaxTextLogits`; KV update is `KvCache::AppendPrefillChunk` /
   `AppendDecodeStep` fed by `CollectKvOutputs`; session counters advance
   only after a step completes. No facade: each step is a distinct call in
   `RunPrefillChunk` / `RunDecodeStep`.
4. **Session policy** → `inc/gemma4_text_session.hpp`,
   `src/gemma4_text_session.cpp`. Pure decisions over
   `TextSessionState {processed_tokens, token_offset, n_keep, history}`:
   `PlanContextShift`, `PlanAutoTruncate`, `PlanContinuationAlignment`
   (the `%kChunkSize` prefix-replay rule), `LastChunkRowIndex`. All four
   mirror the source decision tables exactly, including the give-up branches.
5. **Engine** → `gemma4_text_engine.{hpp,cpp}` becomes the multi-turn
   orchestrator: chunked prefill suffix, decode loop, EOS/callback handling,
   full return-vector semantics, benchmark timing scope, context-shift/
   truncate execution (policy from stage 4, cache mutation via `KvCache`),
   golden `ExportPrefillChunk`, and an explicit `SetDebugSink`. The engine
   prints nothing implicitly; when a caller installs a sink it receives the
   same one-line prebuilt-hidden/alignment diagnostics the library used to
   write to `std::cerr` under `GEMMA4_DEBUG`. Library code keeps no
   environment-variable printing.

### Preserved contracts (each asserted by tests)

Greedy argmax with `kLogitScale` and first-max ties; EOS set {1, 106};
stream callback `false` stops generation and the prefix already generated is
still returned; the full return vector is always `full_ids + generated`;
image-token embedding substitution and prebuilt-hidden override; prefix
continuation (only the suffix is prefilled; boundary alignment replays
`processed % kChunkSize` tokens through a context shift); `ContextShift`
retains the leading prefix via `CompactShift`; `AutoTruncate` decision
table; benchmark timing scope (load / prefill / decode, warmup outside the
timed window, EOS break, `tokens_per_sec` formula); KV rolling, alias and
prefix-retention semantics; all tensor descriptor/ownership fixes from the
accepted package.

### Compatibility

All five executables (`main`, `gemma4_server`, `gemma4_text_bench`,
`gemma4_demo`, `gemma4_golden_verify`) use only kept public APIs (verified
by grep before implementation: constructor, `LoadMs`, `ResetSession`,
`ProcessedTokens`, `Generate`, `GenerateStream`,
`GenerateWithPromptEmbeddings`, `BuildPromptHidden`, `ContinueGenerateStream`,
`ExportPrefillChunk`, `Benchmark`). No call site changes.

### Test plan

Behavior-focused SDK doubles (no vendor ABI): pure policy tests; stage tests
(exact prepared vectors, strided write bytes, single inference per step,
validated KV rows, failure ownership); a full engine flow suite (generation
tokens, continuation/reset, EOS and callback stops, image embeddings,
malformed inputs, failure ownership/session reuse, benchmark scope, sink and
no-implicit-output checks); and the README stage/session example compiled
against the doubles asserting the exact output lines documented in both
languages. Wired through `tests/native/CMakeLists.txt` and a Python host
wrapper.

## Implementation notes

- `TextEngine` shrank from one 640-line translation unit to an orchestrator
  over the three stage modules plus the session policy module; the public
  class surface is unchanged apart from the added `SetDebugSink`, and
  `ProcessedTokens`/`KeepTokens`/`GetHistory` now read the session state.
- The implicit `GEMMA4_DEBUG` → `std::cerr` print inside input preparation is
  gone from the library. The engine emits the same two diagnostics
  (prebuilt-hidden use, continuation alignment) through a caller-installed
  sink; `main`'s own application-level `RuntimeDebugEnabled` logging is
  unchanged, and the README debug-flag sentence was corrected in both
  languages (Vision `[VLM-FIX]` logging still follows the environment
  variable — untouched scope).
- One behavior improvement was driven by a host test, not silently introduced:
  `WriteBatchInputs` validates the prepared batch's element counts against the
  subgraph shape before writing. The former raw-buffer path passed sizes
  implicitly (stack arrays sized by `seq`), so a malformed prepared context
  could only corrupt memory; the stage boundary makes the check possible and
  the stage test requires it.
- The accepted tensor/ownership packages are intact: `gemma4_text_tensor`,
  `gemma4_model_io` and `gemma4_kv_cache` sources are unchanged by this
  package; the engine now calls them through the transport stage.
- Accepted-package test targets (`text_tensor_flow_test`,
  `text_resources_test`) gained the new stage sources in their build sets —
  required because they compile the same production engine; their assertions
  are unchanged and they still pass.

## Verification

All commands from the repository root; Python is
`../rdk_model_zoo/.venv/bin/python`, CMake/CTest the bundled
`../.coordination/native-build-tools/cmake/data/bin/` tools, sanitizer build
the reused `../.coordination/gemma-vision-sanitized` directory (ASan/UBSan,
reconfigured for the new targets). Logs and hashes in
[evidence](evidence/2026-09-28-gemma-text-stages-remediation/):
`ctest.log`, `unittest.log`, `contract.log`, `source-hashes.txt`,
`checks.json`.

- Native: **19/19 CTests pass** under ASan/UBSan — the 15 pre-existing
  entries (vision, KV, ownership, tensor contract/flow, R2 adoption) plus
  `text_session_test`, `text_stages_test`, `text_engine_flow_test` and the
  executed `readme_text_stages_example`.
- Gemma sample host suite: **30/30 unittests pass** (`test_cpp_text_stages.py`
  compiles all four new native tests from production sources and asserts the
  README example's exact three-line output).
- Sample contract checker (gemma scope): 0 violations / 0 skips / 0
  exemptions; bilingual README parity for the new sections.
- Behavior checks recorded above: exact generation tokens
  (`{11,22,33,44,55,104,100,100}`), callback stop keeps the generated prefix,
  aligned/unaligned continuation (including the source's down-alignment
  replay), image embeddings, malformed inputs leave the session usable,
  inference failure propagates with buffers owned and the session reusable,
  benchmark scope unchanged, sink receives diagnostics only when installed,
  and stderr stays empty otherwise.

## Residual issues and boundaries

- Host SDK doubles are not vendor ABI, real HBM descriptors, or model
  numerics; fixture token ids are patterns, not model outputs. Board evidence
  remains not-run.
- The source's continuation alignment aligns DOWN to the chunk boundary, so
  an unaligned prefix can be discarded and replayed wholesale (up to 255
  tokens). That is preserved source behavior, now covered by explicit tests;
  changing it would be a separate decision.
- The engine keeps per-call prepared contexts as fresh vectors; the former
  cached decode scratch buffers were dropped (allocation cost is negligible
  next to inference). If a board profile shows otherwise, the cache belongs
  inside stage 1, not the engine.
- `main`/`gemma4_server`/`gemma4_text_bench`/`gemma4_demo`/
  `gemma4_golden_verify` were not recompiled here (board SDK and
  tokenizers-cpp required); they use only kept public APIs, verified by grep.
- H7 and global closure stay reviewer-owned; this package does not claim
  them. Vision diagnostics (GEMMA4_DEBUG) were deliberately left unchanged.

## Author follow-up — TEXT-R1/R2/DOC-R1 remediation (after Codex independent review)

Codex's independent review (`2026-09-28-gemma-text-stages-independent-review.md`,
changes required) reproduced two public-API memory errors against this
package. Both were reproduced here first against the materialized pre-fix
sources (`evidence/2026-09-28-gemma-text-stages-remediation/r3-before-fix/`,
`gemma4_text_inputs.cpp` `7b2a6155…`), using the reviewer drivers unmodified:

- **TEXT-R1** — `PrepareBatchInputs(embeddings, {11}, 4096, 1, nullptr,
  kChunkSize)` drove `cache_col_start` negative and wrote before the mask
  buffer (ASan heap-buffer-overflow WRITE, `gemma4_text_inputs.cpp:42`,
  rc −6; `r3-mask-red-run.log`). Fix: a `ValidateMaskGeometry` contract
  (`0 <= chunk_valid <= seq_len`, `1 <= seq_len <= kCacheLen`,
  `chunk_start + seq_len <= kCacheLen`, overflow-safe signed arithmetic)
  enforced in `BuildFullMask`/`BuildSlidingMask` (plus null-buffer checks)
  and in `PrepareBatchInputs` — which also requires exactly `chunk_valid`
  prepared ids. `PrepareDecodeInputs` rejects decode positions outside
  `[0, kCacheLen)`; `QuantizeMask` rejects negative dimensions and null
  buffers. Requests that do not fit the fixed context throw
  `std::invalid_argument` before any allocation or write — nothing is
  clamped, so attention behavior is never silently changed, and every chunk
  of a valid 4096-window generation (chunk starts 0..3840, boundary
  `chunk_start=3840` with `chunk_valid=256`) still prepares and runs.
- **TEXT-R2** — `ContinueGenerate({11,22}, 1, &vector(1))` read 6144 bytes
  past the vector: `full_hidden` is indexed as whole-prompt rows but the two
  public continuation entry points never checked its extent (only
  `GenerateWithPromptEmbeddings` did). Fix: `ContinueGenerateStream` validates
  at entry, overflow-safe, that `full_hidden` holds exactly
  `full_ids.size() * kHiddenSize` floats — before any session mutation or
  stage read, so a rejected call leaves the session reusable — and
  `PrepareBatchInputs` now takes the hidden as `const std::vector<float>*`
  so the public preparation API can establish its own extent (at least
  `(chunk_start + chunk_valid) * kHiddenSize`). Optional no-hidden behavior
  and valid image-embedding use are unchanged; suffix-only storage is
  rejected rather than accepted or synthesized.

Green: both reviewer drivers, recompiled unmodified from the reviewer
evidence directory (binaries written outside it), print
`rejected: mask geometry is outside the fixed 4096-token context window`
and `rejected: full_hidden size mismatch` with rc 0 under ASan/UBSan
(`r3-mask-green-run.log`, `r3-hidden-green-run.log`).

**DOC-R1**: `Generate`/`GenerateStream` Doxygen now state that the return is
the full sequence (prompt followed by generated tokens);
`ContinueGenerate`/`ContinueGenerateStream`/`GenerateWithPromptEmbeddings`
Doxygen now document the whole-prompt hidden extent, indexing and rejection
contract; the paired runtime READMEs gained the extent/window contract list
in the stage section (EN/CN). The tautological
`assertIn("passed", run.stdout + "passed")` was removed: the README example
asserts its exact documented three-line output, and the other native tests
assert their own `... passed` completion line.

New regression coverage: `text_stages_test.cpp` adds the geometry rejection
matrix (reviewer case, negatives, oversized windows, null buffers,
`3841+256`), the valid boundary window `chunk_start=3840/chunk_valid=256`,
token-count mismatch, hidden-extent shortfall and decode positions
`4096/-1` rejected with `4095` valid; `text_engine_flow_test.cpp` adds both
continuation entry points rejecting a short hidden with zero session
mutation, nonzero-prefix-offset rejection and reuse, a full-size hidden
shown to index whole-prompt rows, and a valid full-window generation
(prompt 4096, chunks 0..3840) whose continuation past the context is
rejected without changing the session.

Rechecks after the fix (logs refreshed in the evidence directory): 19/19
native CTests under ASan/UBSan, 30/30 Gemma sample host tests, Gemma-scoped
contract checker 0 violations, and the compiled README example prints
exactly its documented output (`readme-example-run.log`). The reviewer's
30-test/19-CTest baseline results were reproduced before the fix and remain
green after it; reviewer files were not modified. H7 stays open; acceptance
belongs to Codex.
