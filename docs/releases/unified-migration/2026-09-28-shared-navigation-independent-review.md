# Shared and Agent navigation review — Codex

H8-NAV-R1 requires a bounded documentation correction, not product changes.

1. samples/_shared/README.md still says YOLOE conversion/native migration is
pending in both its English and Chinese text. Current YOLOE independent runtime,
scorer and native reviews accept those host implementations; retain the missing
published S floating-output asset and real SDK/board not-run boundaries. Link the
actual canonical conversion/native guides. The image utility section also says
"all three samples", inherited from early pilots; describe current consumers or
avoid an obsolete fixed count after checking actual imports.
2. skills/README.md lists docs/manifests, docs/release and release layouts but
omits the unified docs/release/{x5,s} active manifests and their distinction from
platform snapshots. inspect_repo/read_catalog already support unified layouts;
explain actual selection and preserve unknown/multiple-manifest behavior.
Do not redefine pack maintenance branch, candidate versions, historical Windows
results or claim behavioral acceptance from host tests.
3. CLAUDE.md groups X3 under platforms/x3/docs/release, but the actual historical
X3 manifest directory is platforms/x3/release and is still a current catalog
input. Correct that exact exception, consistent with platform/source guides.
Do not change X3 adaptation scope or archived X5/S identity.

Scope: only those three docs and author report/evidence. Do not change runtime,
skills instructions/shared copies/versions or install anything. Existing README
commands remain unchanged. Codex separately reviews actual skill behavior; fresh
structural checks are not that evidence.

## Fresh deterministic checks

Codex reran reference sync (no drift), pack validator (seven skills, 83 eval
definitions, behavior_evaluated=false), and all 57 skill tool/resource tests.
inspect_repo correctly reports the active integration ref, dirty worktree,
unified X5/S manifest locations, historical X3 location and no inferred hardware.
read_catalog is additionally exercised with an ambiguous implicit manifest and
an explicit S manifest for efficientnet. Full output and helper source hashes
are in `evidence/2026-09-28-shared-navigation-independent-review/tools-and-context.json`.
These checks certify deterministic tools/resource structure only; current-candidate
Agent behavior remains a separate H8 acceptance requirement.

## 2026-09-29 independent follow-up

H8-NAV-R1 accepted in [the document-remediation review](2026-09-29-document-remediation-independent-review.md). Original findings above are retained as history; the new review binds the corrected files and evidence. Broader rollup status is tracked separately.
