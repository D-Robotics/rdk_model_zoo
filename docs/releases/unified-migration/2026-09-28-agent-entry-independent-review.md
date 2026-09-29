# Agent entry independent review — changes required

Reviewer: Codex. Base `8876f379`; this finding concerns the tracked CLAUDE.md,
not installed skills or automatic memory. H8 remains open.

## ENTRY-R1 — stale migration instructions contradict the live contracts (P2)

CLAUDE.md lines 15–19 claim skills are absent and direct validation away from
this checkout. The tracked skills pack, its tests and tools already exist.
Line 53 presents archived platforms manifests as active until a future A4;
AGENTS.md and production asset resolution already use docs/release/{x5,s}.
Lines 29–42 impose Python everywhere, the historical dict/tuple interfaces,
utils/py_utils and a universal constructor/CLI shape. The accepted inference
contract instead defines explicit per-call context, typed results, separated
binding/runner/task modules, sample-specific native APIs and justified streaming
interfaces. Native MiniCPM/Gemma are concrete counterexamples to universal Python.
The release paragraph also repeats S/X3 delivery ordering although X3 is now
archive-only. A generic spec-precedence statement does not make these concrete
instructions useful to the next implementing Agent.

Required bounded correction: rewrite the active checkout guidance using the
current AGENTS.md, docs/sample-standards/{inference,readme}-contract.md and ADRs.
Keep useful repository navigation and supported host commands; identify source
branch conventions as historical references where needed. Point to authoritative
contracts rather than duplicating long prescriptions that can drift. Explain the
current integration branch versus customer delivery branches; do not imply this
workbranch has merged into develop or is released. Preserve the user's trusted
quantization-document scope and distinct host/board evidence. No product runtime
change, installed-skill change, memory update, quantization or board run is needed.

Acceptance: statically resolve local paths, compare the revised instructions
against the live directory/API and current scope, and have Codex independently
review the diff. The separate skills increment candidate is still in progress;
its author completion must not be described as release or whole-H8 acceptance.

## Independent closure — ENTRY-R1 accepted (2026-09-28)

Codex inspected the final CLAUDE.md diff after the implementation session exited.
It now identifies active manifests and the local skills tree, points to current
README/inference contracts, describes sample-specific language coverage and
retains the archive-only X3 boundary. The current integration workbranch is
explicitly distinct from customer delivery/release state. Local literal paths
resolve and absent-path statements match the tree; see
[static-check.json](evidence/2026-09-28-agent-entry-independent-review/static-check.json)
for the accepted file hash, base and checks. The trusted quantization-document
scope is inherited through its required AGENTS.md entry, not replaced by a new
requirement to execute recipes. No product code or installed-memory file changed
as part of this acceptance.

The author's report carries the earlier inherited session base and its then-open
skills-review status. Acceptance here is bound to the actual final CLAUDE.md hash;
the upstream skills increment has separately been accepted in 6c2bdceb. Neither
statement establishes whole-H8/branch acceptance or a release. This migration-era
branch-status paragraph must be updated when actual integration/release occurs;
current scoped sample acceptances remain distinct from whole-branch acceptance.

No runtime tests were added or repeated for this text-only change. ENTRY-R1 closes;
whole-H8 Agent navigation and final delivery review remain open.
