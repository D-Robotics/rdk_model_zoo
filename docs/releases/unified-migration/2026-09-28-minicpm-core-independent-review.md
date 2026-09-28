# MiniCPM core independent review — changes required

Reviewer: Codex. Reviewing the uncommitted package described in
`2026-09-28-b11-minicpm-core-refactor.md`, integration base `e7a76f58`.
Independent host run: all 16 author tests pass. This does not cover the two
reproduced blockers below. [Evidence](evidence/2026-09-28-minicpm-core-independent-review/)
contains the full independent suite output, pointer driver, compilation/run record,
manifest parse output and candidate hashes. No board, real SDK or model was run.

## CORE-R1 — request storage has dangling self-pointers after return/move (P1)

`runtime/legacy/inc/minicpm5.hpp:30–34` stores strings and the SDK request/input
alongside pointers into those same members. `src/minicpm5.cc:65–70` fills those
pointers then returns a named local. Default copy/move does not rebind them.
C++17 does not require named return-value optimization. The independent driver
compiled real production sources against the author's explicitly fake xlm header
with `-fno-elide-constructors`; return/copy/move ownership checks all report false,
exit 1. Even normal NRVO does not make subsequent public copies/moves safe.

Use stable ownership, correctly implemented special members, or construct the
SDK pointer view only immediately before the call over stable owned storage.
Document which operations the carrier allows. Cover non-elided return, short and
long strings, and all allowed copy/move operations. Do not merely turn on an
optimization flag or disable the failing case. Preserve streaming and lifecycle.

## CORE-R2 — manifest note makes every S asset lookup fail (P1)

`docs/release/s/models.yaml:1898` contains unquoted `path: the B11` inside a plain
multiline scalar. PyYAML rejects it with `mapping values are not allowed here`.
This is a repository-wide S manifest regression, also causing the concurrently
running PointNet suite to fail. Fix the scalar syntax first, preserving every
URL/hash/asset and intended note. Verify YAML parsing and a real manifest-backed
selection after correction. The README checker is not a manifest validator;
its zero violations cannot close this defect.

## CORE-R3 — streaming console output still lives in inference (P2)

`runtime/legacy/src/minicpm5.cc:20–30` hardcodes std::cout in the model callback.
The agreed package specifically required console/report output outside the core.
Move console presentation to main via an injected streaming sink/callback or an
equivalent explicit consumer. Preserve incremental token delivery and suppression
of END/error text, status mapping, and cleanup. The asynchronous-callback comment
does not require console IO in this file. Cover a custom sink and the default CLI,
including consumer failure handling without throwing across the vendor callback.
LLM streaming interfaces are allowed; no artificial tokenizer API is requested.

## Documentation and public API follow-through

- Runtime README banners still say this round only covers launcher orchestration.
  Update current host-refactor/test scope while retaining historical board claims.
- Explain the public request/storage lifetime, sequential usage and streaming sink
  through a complete native library usage example. Existing CLI commands remain.
- The public S600 pre_process accepts max_new_tokens/request_id directly while
  relying on constructor validation that direct callers bypass. Either make helpers
  internal or enforce/document their actual public argument contract. Do not claim
  constructor checks cover unrelated public calls.

Original baseline lifecycle, metric and temporary-file fixes are represented by
passing tests and should be retained. Discovery counts and links are updated to
51 native samples; archive identity remains unchanged in the diff. Original R5
cannot close while its note breaks YAML. Whole MiniCPM acceptance remains
changes-required; do not advance or mark B11/H7 complete from the 16 green tests.
