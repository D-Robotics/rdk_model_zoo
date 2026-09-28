# Catalog integration — independent host review

Reviewer: Codex. Baseline f50585de. Status: source/build checks accepted;
H8-CATALOG-DOC-R1 remains open pending author README correction.

Ran the complete publisher `npm run check` using the existing Node v22.23.2
runtime and installed dependencies: source validation, 121 tests in 18 files,
TypeScript checking, catalog build, and checksum/reproducibility verification
all passed. No packages were installed. [Full output](evidence/2026-09-28-catalog-independent-review/check.json)
retains warnings: 125 X5, 35 S, and 18 historical X3 accuracy metrics have no
published dataset. These are incomplete source metadata, not synthesized data.

The emitted identity is catalog-v1.0.0-bcb085e3cb057c77, with 57 catalog sample
cards, 567 assets (565 downloadable) and 812 benchmark records. These counts
include historical X3 and catalog family aggregation; they are not native
migration-completion counts. [Metadata snapshot](evidence/2026-09-28-catalog-independent-review/catalog.meta.json)
records payload SHA-256 and bytes, independently recomputed from the local
artifact. [Reviewed input hashes](evidence/2026-09-28-catalog-independent-review/reviewed-input-hashes.json)
bind source/configuration/test/manifest material. Generated dist files remain
local build products, not a published release.

## H8-CATALOG-DOC-R1 — Incorrect maintenance source directions

The publisher README still states that all current data is maintained under
platforms/x5 and platforms/s on main. Actual sources.json reads the working
tree root at docs/release/x5 and docs/release/s, with per-platform VERSION
files and generated link_ref develop. Only X3 remains platforms/x3/release
with main links. The README therefore directs maintainers to archived source
locations and confuses worktree input with link destinations.

Assigned a documentation-only correction to Claude Code + GLM, queued after
the dataset package. The explanation must also preserve the actual CI scope:
main pushes, matching PR paths and manual dispatch; the present integration
branch push is not website publication or proof that develop contains its files.
Historical tag overrides and separate documentation-site import remain intact.
No workflow, source configuration, default branch, tag or external site was changed.
H8 remains open for this correction, dataset documentation and remaining shared
integration review.

## H8-CATALOG-R2 — Historical tag pin inherits worktree VERSION path (P2)

The advertised `npm run catalog:build -- --pin x5=x5-v1.1.2 --out ...` fails under Node 22.23.2 (rc=1): `git show x5-v1.1.2:docs/release/x5/VERSION` returns 128 because that immutable historical tag has a root VERSION. `resolvePlatformSources` inherits current `entry.version_file` even when `--pin` switches to a different historical layout. Manifest directory probing finds old `docs/release`, but VERSION is not resolved alongside it. Evidence: `historical-pin-failure.json` records full command/output and source hashes. The earlier full current-worktree check passed; it did not exercise this advertised historical-pin command.

Preserve historical annotated-tag capability and current unified worktree validation. Resolve VERSION using the selected immutable tag/layout, handle prefixed tag trees, and reject genuinely absent or mismatching versions. Add behavior tests for old root layout, prefixed legacy layout and unified tags while retaining strict worktree version checking; independently rerun the exact advertised historical example. No tag rewriting, manifest asset changes, release or site publication. Assigned sequentially to the catalog Claude Code + GLM session after its current doc-only task. H8 remains open.

## Final independent recheck — Catalog package accepted

Codex reviewed the complete sources.ts/test/helper diff and final publisher
README against sources.json, the registry tests and the workflow. Version lookup
now uses the manifest directory resolved first; tag tree prefixes apply to both
files, and worktree VERSION reads remain strict on the configured path. Tests
cover historical root, prefixed legacy, unified, missing/mismatch and fallback
colocation with a stale root version. No manifest/tag/asset change is involved.

Fresh `final-recheck.json` records all commands and unmodified candidate hashes:
the original two isolated reviewer cases both report correct=true; full npm check
passes source validation, 130 tests across 18 files, TypeScript, build and checksum
verification. The exact advertised x5-v1.1.2 pin builds successfully and a second
pinned check proves reproducibility. Existing dependencies and Node22 were used,
with no installs, model downloads, real conversion or board access.

Current artifact remains catalog-v1.0.0-bcb085e3cb057c77 (57 families,
812 benchmarks); pinned historical content is catalog-v1.0.0-89ed4880524e4787
(56 families, 800 benchmarks). `final-catalog-metadata.json` retains both metadata
objects; Codex independently matched their payload SHA-256 and byte lengths.
Historical X3/source-metadata warnings are retained, not repaired with invented
measurements. Catalog counts are not sample migration acceptance counts.

The README now correctly distinguishes sources.json inputs, generated develop
links, default worktree reads, optional historical tag prefixes and actual CI
upload triggers. **Close H8-CATALOG-DOC-R1 and H8-CATALOG-R2**, including the
VERSION fallback follow-up. All original failures and author round-1/round-2
records remain preserved. Parent platform README wording is tracked separately
as ENTRY-DOC-R1; broader H8/shared/skills/navigation and H9 remain open. No release,
website, tag or develop merge was performed.
