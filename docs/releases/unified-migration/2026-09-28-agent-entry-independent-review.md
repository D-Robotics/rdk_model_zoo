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
