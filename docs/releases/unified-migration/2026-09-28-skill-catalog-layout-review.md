# Skill catalog discovery — independent finding H8-SKILL-R1

Codex ran the documented read_catalog.py --repo <actual checkout> --model
efficientnet. It returns rc=2, reason=manifest-not-found, despite the active
unified manifests under docs/release/x5 and docs/release/s. Explicit
--manifest docs/release/s/models.yaml succeeds. inspect_repo correctly discovers
both active manifests plus historical X3. Full real-command evidence is in
`evidence/2026-09-28-shared-navigation-independent-review/tools-and-context.json`.

read_catalog only probes flat docs/manifests, docs/release, release. Its default
invocation therefore gives a false absence instead of requiring target selection.
The 57 existing tests pass but do not cover default discovery on this layout.

Preserve explicit path selection and historic single-platform layouts. Add
known unified/per-platform discovery with active manifests preferred over archived
snapshots for the same platform, consistent with inspect_repo. When multiple
platform manifests exist, report ambiguity and actionable candidate paths; never
infer target from branch, filename or the S group. Missing manifests remain
unknown. Keep standalone single-skill installation self-contained: do not import
a sibling skill or repository-only shared helper. Retain path traversal/YAML
limits and no network/download/runtime behavior. Add real regressions for unified
multi-platform, one available manifest, active-vs-snapshot precedence and missing.

Scope: read_catalog.py, appropriate skills tests and user-facing usage/help;
version/governance metadata only if required by pack rules, documenting unreleased
candidate status. No published tags/installations/Hub changes. Existing upstream
corrections and Q5 constraints must remain intact. Codex independently reviews.

## 2026-09-29 independent closure

H8-SKILL-R1 accepted after actual-checkout ambiguity/explicit-selection checks and 63 host tests. See [current Skills disposition](2026-09-29-skills-behavior-independent-review.md) and its source-bound evidence. The original failure above is retained.
