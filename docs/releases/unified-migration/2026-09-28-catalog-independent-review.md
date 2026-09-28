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
