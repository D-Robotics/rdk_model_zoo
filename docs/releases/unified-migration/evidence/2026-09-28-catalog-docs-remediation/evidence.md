# Evidence — H8-CATALOG-DOC-R1 catalog-publisher README correction (2026-09-28)

Implemented by Claude Code + GLM; pending independent Codex review. No H8 closure
is claimed here. All verification is static source/README comparison; no npm,
no dependency install, no build, no network (Codex runs `npm run check` separately).

## 1. Scope proof

`git status --porcelain tools/catalog-publisher` → exactly
`M tools/catalog-publisher/README.md` (see `check_claims.py` §9, output
`check_claims.out`). New files exist only in
`docs/releases/unified-migration/2026-09-28-catalog-docs-remediation.md` and this
evidence directory. No script, config, workflow, manifest, sources.json or
generated `dist/` file was touched.

Hashes:

- `source-hashes-before.txt` vs `source-hashes-after.txt`: same six-file
  SHA256 list before and after; the only differing line is the README
  (`53f27fa2…` at HEAD → `a2f8e54f…` now). `sources.json` (`ff4d5e95…`), the
  workflow (`5a493cee…`) and all three VERSION files are byte-identical
  across the change.
- Before text: `readme-before.md` (verbatim HEAD capture);
  `sources-at-head.json` (sources.json capture, hash matches both captures).
- Full diff: `readme.diff` (3 lines removed, 17 added; all other sections
  byte-identical — asserted programmatically, §3).

## 2. The finding, verified against sources

Old README (captured verbatim in `readme-before.md`) claimed:

> All current platform sources are read from `main` through `sources.json`:
> `platforms/x5`, `platforms/s`, and `platforms/x3`. New models, fixes and
> measurements are maintained there.

Actual `sources.json` (captured at `sources-at-head.json`, hash unchanged after):

| Platform | mode | path | manifest_root | version_file | link_ref | link_prefix |
| --- | --- | --- | --- | --- | --- | --- |
| x5 | worktree | `.` | `docs/release/x5` | `docs/release/x5/VERSION` | develop | (empty) |
| s | worktree | `.` | `docs/release/s` | `docs/release/s/VERSION` | develop | (empty) |
| x3 | worktree | `platforms/x3` | `release` | (default `VERSION` → `platforms/x3/VERSION`) | main | `platforms/x3` |

Supporting sources (line refs at HEAD):

- `tools/catalog-publisher/src/sources.ts:118-123` — `worktree` sources are read
  with plain filesystem `readFile` against the checked-out tree; only `tag`
  sources/pins use `git show`.
- `src/sources.ts:36-39` — `link_ref` is documented as "Git ref used for
  repository source links (`blob/<ref>/...`)", i.e. link labeling, not read
  selection.
- `src/catalog/variants.ts:644-645` — records are emitted with
  `source_ref: options.sourceRef ?? releaseTag` /
  `source_path_prefix: options.sourcePathPrefix`.
- `src/pipeline/multiplatform-catalog.ts:393,397` — provenance `ref` comes from
  `source.linkRef`; `manifest_sha256` digests the manifests actually read.
- `src/pipeline/multiplatform-catalog.ts:59` — build rejects a `VERSION` that
  disagrees with the manifest `release.version` (current tree agrees:
  x5 1.1.3, s 1.1.2, x3 1.1.2).
- `tests/platform-registry.test.ts` — `platforms/registry.json` is only
  cross-checked against `sources.json` by tests; no code under `src/` or
  `scripts/` reads it (grep-verified, `check_claims.py` §4 sources.ts/variants
  checks; the registry file itself is untouched).
- `.github/workflows/model-catalog-data.yml:3-19` — triggers are
  `pull_request` (path-filtered), `push` restricted to `branches: [main]`
  (path-filtered), and `workflow_dispatch`; step 57 `if:
  github.event_name != 'pull_request'` uploads exactly
  `dist/catalog.json` + `dist/catalog.meta.json` on non-PR runs only.
- Generated-artifact confirmation (pre-existing `dist/`, not rebuilt):
  `dist/catalog.meta.json` provenance records `ref: develop` / `develop` /
  `main` for worktree reads, with 64-hex `manifest_sha256` per platform —
  demonstrating that the emitted ref is a link label while the bytes come from
  the worktree.

## 3. Verification

`check_claims.py` (run from repo root; output `check_claims.out`):
75 checks passed, 0 failed. Coverage:

1. Old false paragraph removed; worktree-mode statement present.
2. Per-platform sources.json fields match the README table; all read files
   exist in the worktree; each VERSION agrees with its manifest
   `release.version`.
3. Archived snapshots `platforms/{x5,s}/docs/release/models.yaml` exist but are
   referenced nowhere in sources.json; README states both facts.
4. sources.ts/variants.ts/multiplatform-catalog.ts mechanics back the new
   "Worktree reads versus generated links" section.
5. `dist/catalog.meta.json` provenance refs/kinds/digests agree with the README
   description.
6. Workflow triggers, non-PR-only upload, two-file artifact set, and the
   README's CI paragraph agree; separate-publication boundary sentence kept
   verbatim.
7. Preserved sections byte-identical to the before capture: Node requirement,
   npm script block, "Data identity and historical releases" (incl. annotated
   tag pins and the `--pin x5=<annotated-tag>:platforms/x5` override), docs
   import block, and the closing no-synthesis rule.
8. README relative links resolve (`../catalog-redirect` exists).
9. Scope: only `tools/catalog-publisher/README.md` modified in the module.

Note on checker history: an intermediate run showed 2 failures caused by a bug
in the checker itself (`.strip()` removed porcelain's leading status space,
misreading ` M` as staged `M `). Fixed to `rstrip("\n")`; the failure never
reflected the repository state. This is recorded so the reviewer can trace the
intermediate output if they saw it.

## 4. Deliberate decisions for review

- No `README_cn.md` added (task marked it optional): all sibling tool READMEs
  (`tools/sample_contract/`, `tools/catalog-redirect/`,
  `tools/board_validation/`) are English-only; the bilingual-pairing rule of
  `docs/sample-standards/readme-contract.md` governs sample-level READMEs.
  Adding a pair would create a standing sync obligation for the only tool docs
  that have one.
- The "Adding hardware" sentence "does not require changing the main branch"
  was rephrased to "does not require moving the dashboard or relocating the
  maintained manifests": same meaning, without feeding the removed
  main-branch confusion.
- `dist/` was not rebuilt or modified; `catalog.meta.json` was read as
  evidence only. Reproducibility of the artifact is Codex's separate
  `npm run check` scope.

## 5. Out-of-scope observations (not edited)

- `platforms/README.md:18` (+ `README_cn.md:18`) says the publisher "locates
  them through the registry" (`从注册表定位这些清单`); the build actually
  resolves from `sources.json` and only tests cross-check the registry. The new
  README states the mechanism precisely; the parent README phrasing was left
  untouched (outside allowed writes).
- `platforms/x5/docs/release/README.md:33` still instructs running the check
  "on the aggregated `main` branch" with the manifest at
  `platforms/x5/docs/release/` — frozen delivery-branch material describing its
  own historical layout; consistent with the archive contract, left untouched.
- Root `README.md` §"Catalog data (maintainers)" (`README.md:107-116`,
  `README_cn.md` same) is consistent with the corrected README (Node 22.12+/<23,
  same commands, "CI uploads data artifacts", "Catalog data, website publication
  and board samples are separate deliveries").
