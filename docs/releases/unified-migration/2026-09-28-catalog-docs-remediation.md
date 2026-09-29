# H8 (bounded): catalog-publisher README source-description correction — 2026-09-28

Status: **implemented by Claude Code + GLM; pending independent Codex review. H8 is not closed by this document.**

Scope: correct `tools/catalog-publisher/README.md`, whose "Local development"
section described the catalog sources as read from `main` at
`platforms/x5`/`platforms/s`/`platforms/x3` with new models/fixes maintained
there. The actual `sources.json` reads the worktree: unified manifests
`docs/release/{x5,s}` for the active platforms and the archived
`platforms/x3/release` for the historical X3 distribution. Documentation-only:
no script, config, workflow, manifest, `sources.json` or generated `dist/`
output was modified; no npm install, build, board run or network access was
used (Codex runs `npm run check` separately).

## Finding addressed

| Finding (review input) | Resolution |
| --- | --- |
| H8-CATALOG-DOC-R1: README claimed all sources read from `main` under `platforms/*`, directing maintenance to the wrong location | New "Data sources" section: per-platform table of the real `worktree` reads (`docs/release/{x5,s}` manifests + per-platform `VERSION`; `platforms/x3/release` + `platforms/x3/VERSION`); unified manifests named as the maintenance location; `platforms/{x5,s}/docs/release/*.yaml` explicitly described as archived frozen delivery-branch snapshots that `sources.json` never reads |
| Working-tree reads versus generated `link_ref` conflated | New "Worktree reads versus generated links" subsection: `link_ref`/`link_prefix` only label emitted `source_ref`/`source_path_prefix` and provenance `ref`; content always comes from the checked-out tree; an integration-branch build emits `develop` links without `develop` containing those candidates until merge; per-platform `manifest_sha256` records what was actually read |
| CI trigger claim vague enough to imply workbranch pushes publish | CI paragraph now names the actual triggers (`pull_request` path-filtered, `push` to `main` only path-filtered, `workflow_dispatch`), states the two-file artifact upload happens only on non-PR runs, and states that integration-workbranch pushes trigger no workflow; the separate-enabling / separate-website-publication boundary sentence is preserved verbatim |
| Preserved material (verified byte-identical by `check_claims.py` §7) | Node.js ≥22.12 <23 requirement and exact `npm ci`/`npm run check` block; data-identity section (`catalog-v1.0.0-<fingerprint>`, metadata SHA256, platform tags separate); historical annotated-tag pin instructions incl. `--pin x5=<annotated-tag>:platforms/x5` and the no-tag-rewrite rule; docs-import block and `../catalog-redirect` pointer; closing "no values are synthesized" rule |

## Verification (host, static)

- `evidence/check_claims.py`: 75/75 checks pass (`evidence/check_claims.out`).
  Covers every factual sentence of the rewritten README against `sources.json`,
  `src/sources.ts`, `src/catalog/variants.ts`,
  `src/pipeline/multiplatform-catalog.ts`, the workflow YAML, the three VERSION
  files, `tests/platform-registry.test.ts` (registry is test-only cross-check,
  not a build input), and pre-existing `dist/catalog.meta.json` (provenance
  refs `develop`/`develop`/`main` for worktree reads).
- Preserved sections byte-identical to the HEAD capture
  (`evidence/readme-before.md`); full diff in `evidence/readme.diff`
  (3 lines removed, 17 added).
- Scope: `git status --porcelain tools/catalog-publisher` shows exactly
  `tools/catalog-publisher/README.md` modified; before/after hashes in
  `evidence/source-hashes-{before,after}.txt` show `sources.json`, the workflow
  and all VERSION files unchanged.
- The README's claims that the build rejects VERSION/manifest disagreement and
  emits `blob/<ref>/...` links are backed by
  `multiplatform-catalog.ts:59` and `sources.ts:36-39` respectively; current
  tree agrees on all three versions (x5 1.1.3, s 1.1.2, x3 1.1.2).

## Remaining limits (not claimed)

- No build was run: catalog reproducibility, `npm run check`, tests and
  typecheck are Codex's separate scope; `dist/` is untouched pre-existing
  output used read-only as evidence.
- No statement about any published artifact or website is made or implied; the
  migration prepares local files only, and this work branch's pushes trigger
  no CI workflow.
- `platforms/README.md:18` still says the publisher locates manifests "through
  the registry"; the build actually resolves from `sources.json` (registry is
  cross-checked by tests only). Parent README is outside this task's allowed
  writes — flagged in `evidence/evidence.md` §5 for the reviewer.
- `platforms/x5/docs/release/README.md:33` (frozen delivery-branch snapshot)
  retains its own historical `main`-branch instruction; that is archive
  material describing its tagged layout, not edited here.
- No `README_cn.md` was added (optional): sibling tool READMEs are
  English-only and the readme-contract bilingual-pairing rule governs sample
  READMEs; decision recorded in `evidence/evidence.md` §4.
- H8 sub-item only. H8 overall and H0–H9 remain open; independent Codex review
  of this package is pending.

## Files

- Modified: `tools/catalog-publisher/README.md` (only tracked change)
- New: `docs/releases/unified-migration/2026-09-28-catalog-docs-remediation.md` (this record)
- Evidence: `evidence/2026-09-28-catalog-docs-remediation/`
  (`evidence.md`, `check_claims.py`, `check_claims.out`, `readme.diff`,
  `readme-before.md`, `sources-at-head.json`,
  `source-hashes-before.txt`, `source-hashes-after.txt`)
