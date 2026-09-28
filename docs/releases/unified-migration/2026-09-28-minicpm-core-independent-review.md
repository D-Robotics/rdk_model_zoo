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

## CORE-R4 — native test drivers reuse and delete shared temporary paths (P1)

Further review of the R2 candidate found that `s600_metrics.cpp` and
`s600_stages.cpp` create fixtures at `temp_directory_path()/model`, truncate
four files there and remove the entire directory afterward. The cleanup driver
also uses fixed scenario directory names despite claiming concurrent isolation.
These paths are not owned exclusively by the test. Running the drivers directly
can overwrite or delete unrelated files; concurrent test runs can corrupt one
another's fixtures.

Codex compiled the actual metrics driver and production sources, then ran it
with TMPDIR pointing to a newly created reviewer-owned directory containing
`model/reviewer-unrelated.txt`. The driver returned success but removed that
pre-existing sentinel. No external directory was exposed to this reproduction.
[Full command and result](evidence/2026-09-28-minicpm-core-independent-review/test-temp-isolation.json).

Use atomically allocated unique per-driver scratch directories, with RAII
cleanup only of owned paths; restore TMPDIR including its initially unset state
when changed. Preserve this sentinel case and concurrent isolation as meaningful
regressions. Do not merely hide fixed unsafe paths behind the Python launcher:
the native drivers must also be safe when invoked directly. This is an additional
blocking test-harness finding; original CORE-R1–R3 closure awaits final recheck.

## Final recheck — CORE-R1–R4 resolved; README example correction pending

Independent rerun after the R3 author exit: 20 tests pass, including direct
compiled-driver sentinel and concurrent scratch checks. The reviewer also ran
the original non-elided ownership reproducer: return/copy/move all report owned=1.
The manifest parses, all non-note structure matches HEAD, and three actual
MiniCPM assets enumerate correctly. Console streaming now uses the injected sink;
END/ERROR suppression and contained consumer exceptions have explicit tests.
Source SDK exit mapping intentionally remains separate from stream_error.
The cleanup driver additionally passed with TMPDIR absent from its environment
and asserted restoration to that initially unset state. CORE-R1–R4 are resolved
within this host scope. Original failures above remain preserved. Evidence:
[r3 recheck](evidence/2026-09-28-minicpm-core-independent-review/r3-recheck.json),
[ownership/manifest](evidence/2026-09-28-minicpm-core-independent-review/ownership-manifest-recheck.json),
[unset environment and hashes](evidence/2026-09-28-minicpm-core-independent-review/unset-snapshot-recheck.json).

CORE-N1: Both legacy runtime README C++ examples claim complete library usage but
use std::cout/std::flush without including iostream. Extracting their actual
blocks and placing statements in main fails syntax compilation against the
production header and explicitly fake SDK header.
[Actual compiler errors](evidence/2026-09-28-minicpm-core-independent-review/readme-example-compile.json).
Supply self-contained examples in both languages and verify their actual blocks;
check the S600 pair in the same bounded documentation pass. No runtime rewrite or
full-suite rerun is required for this header/example correction. Package acceptance
awaits that customer documentation correction; B11/H7 remain globally open.

## Final disposition — MiniCPM core package accepted within host scope

CORE-N1 is resolved. After Claude exited, Codex extracted each complete C++ block
from the two runtime guides in both languages and syntax-compiled it unchanged
with C++17 -Wall -Wextra -Werror against production headers and the existing
explicit SDK doubles. All four pass.
[Full example sources and compiler results](evidence/2026-09-28-minicpm-core-independent-review/readme-example-recheck.json).
Hash comparison with the independently tested core snapshot confirms only those
four README files changed; the 20-test code baseline remains identical.

Accept the current MiniCPM core refactor and CORE-R1–R4/CORE-N1 remediation.
The original failed review and reproductions are retained above. Root/index
integration adds the previously missing MiniCPM link and updates the native
inventory to 51; the S manifest change is notes-only with asset identity intact.
Author timeline entries are historical self-reports, superseded by this scoped
independent disposition. Keep the source S100/S100P precision failure
(PPL +27.83%, 2/6 reference matches) and S600 historical +1.60% distinct.
No live board, vendor ABI, model precision or performance acceptance is claimed.
Gemma Text, B11 as a batch, H8/H9 and whole-branch acceptance remain open.
