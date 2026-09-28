# Catalog VERSION fallback follow-up — independent review

Reviewer: Codex. H8-CATALOG-R2 remains changes-required.

The initial fix handles the advertised historical root VERSION, but the new
resolver receives the original source object, not the resolved manifestDirectory.
At sources.ts resolvePlatformSources, object-literal field evaluation does not
mutate source. Thus a tag selected via fallback `release/models.yaml` searches
`docs/release/x5/VERSION` and root `VERSION`, ignoring `release/VERSION`.

Two isolated tagged fixtures reproduce this (no real repository tags changed):
1. Valid `release/VERSION` only: falsely rejects with no VERSION found.
2. Same valid colocated file plus stale root VERSION: selects root VERSION.
See `evidence/2026-09-28-catalog-independent-review/version-fallback-counterexample.json`
and its reproducible `.mts` script. Both results report correct=false.

Required correction: resolve manifest directory first, then resolve VERSION
using that actual directory. Cover both cases, preserve strict worktree behavior
and mismatch rejection, rerun historical pin and current npm check. Product fix
belongs to Claude Code + GLM.

README also still calls `platforms/x5` the future/new layout, contrary to the
current root `docs/release/x5` layout. Describe the optional tree prefix only for
tags actually carrying that prefixed tree. Qualify the worktree-only statement
that a build always reads worktree so it cannot contradict explicit tag pins.
