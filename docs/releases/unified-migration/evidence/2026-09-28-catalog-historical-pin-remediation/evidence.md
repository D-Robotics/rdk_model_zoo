# Evidence — H8-CATALOG-R2 historical-pin VERSION resolution fix (2026-09-28)

Implemented by Claude Code + GLM; pending independent Codex review. No H8
closure is claimed. The Codex review report and its
`evidence/2026-09-28-catalog-independent-review/` directory — including
`historical-pin-failure.json` (sha256 `1092b553…`) — are preserved unmodified.

**Round 2 (version-fallback sequencing) additions** — files with suffix
`round2` plus `codex-reproducer-after.log` and `sources-ts-cumulative.diff` /
`sources-test-round2-cumulative.diff` / `readme-cumulative.diff` /
`hashes-after-round2.txt`; see the follow-up section of
`docs/releases/unified-migration/2026-09-28-catalog-historical-pin-remediation.md`.
The author's pre-fix run of Codex's two-case reproducer observed both cases
`correct=false` (matching the preserved `version-fallback-counterexample.json`,
sha256 `efededee…`); the post-fix author run is `codex-reproducer-after.log`
(both `correct=true`). Round-1 files below are unchanged.

## 1. Reproduction (before fix)

`npm run catalog:build -- --pin x5=x5-v1.1.2 --out <dir>` with the existing
Node v22.23.2 at `/opt/homebrew/opt/node@22/bin`, no installs
(`reproduce-before-stdout.log`, `reproduce-before-stderr.log`):

- rc=1; same failure Codex recorded: `git show
  x5-v1.1.2:docs/release/x5/VERSION` → 128, "path … exists on disk, but not in
  'x5-v1.1.2'".
- Root cause confirmed in `src/sources.ts` at HEAD `c580b6d9…` — the exact
  hash Codex's `historical-pin-failure.json` records for the file, so the
  reproduction matches the reviewed state byte-for-byte:
  `resolvePlatformSources` kept `entry.version_file` (the current worktree's
  `docs/release/x5/VERSION`) for pin/tag sources, while the manifest directory
  probe correctly followed the tag's own layout (`docs/release`).
- The x5-v1.1.2 tag layout (read-only `git ls-tree`): root `VERSION` (1.1.2),
  manifests at `docs/release/models.yaml` + `benchmarks.yaml`. The x3-v1.1.2
  tag has root `VERSION` only and `release/` manifests — which is why the x3
  pin (default `version_file` "VERSION") never exposed the leak.

## 2. Fix (`sources-ts.diff`, after hash `86b166ae…`)

In `resolvePlatformSources` (tools/catalog-publisher/src/sources.ts):

- New `resolveVersionFile`: worktree sources return the configured path
  unchanged — strict, no probing, missing file still fails the build (existing
  behavior preserved). Tag sources (pins and `mode: "tag"` entries) resolve
  VERSION from the layout the selected ref actually carries, probing in order:
  1. `<resolved manifest directory>/VERSION` (unified colocated layout),
  2. `VERSION` at the platform root (legacy layout; `joinTreePath` applies the
     pin's tree prefix, covering prefixed tags).
  No candidates from the worktree configuration are consulted — the
  configured `version_file` describes the current checkout and was the leak
  vector.
- A tag carrying manifests but no VERSION fails with a shaped error
  `x5: no VERSION found under <tag> (tried …)` instead of a raw `git show`
  exit 128, mirroring `resolveManifestDirectory`'s error style.
- Version/manifest disagreement continues to be rejected downstream in
  `manifestPair` (multiplatform-catalog.ts "VERSION … disagree"), now over the
  file the tag layout actually selected.
- Both resolution steps run for every source symmetrically; `version_file`'s
  doc comments now state it applies to worktree mode only.

Invariants held: platform source identity (`PLATFORMS`, `SourceEntry` schema,
`sources.json`) untouched; emitted `source_ref`/`source_path_prefix`,
provenance, asset URLs and hashes derive from `linkRef`/`linkPrefix` and the
manifest content — all unchanged for worktree builds.

## 3. Regression tests (`sources-test.diff`, new helper `tag-repository-helper-new.ts`)

`tests/sources.test.ts` grows from 7 to 14 tests (suite total 121 → 128). New
`pinned platform VERSION resolution` cases:

1. `--pin x5=x5-v1.1.2` resolves `manifestDirectory "docs/release"`,
   `versionFile "VERSION"` from the real tag (the R2 regression).
2. The advertised pin performs a full `buildMultiplatformCatalog`:
   `platform_tags.x5 === "x5-v1.1.2"`, all x5 variants carry
   `source_ref "x5-v1.1.2"`.
3. Prefixed legacy layout (synthetic isolated git fixture:
   `platforms/x5/{docs/release/{models,benchmarks}.yaml, VERSION}`): manifests
   resolve under the prefix, VERSION resolves to the prefix root, emitted
   links keep `link_prefix platforms/x5`.
4. Unified-layout tag (synthetic `docs/release/x5/{models,benchmarks,VERSION}`):
   colocated VERSION selected.
5. Tag with manifests but no VERSION anywhere: rejected
   `/no VERSION found under x5-v9.9.9 (tried docs/release/x5/VERSION, VERSION)/`.
6. Tag whose VERSION disagrees with its manifest identity: build rejected
   `/VERSION 0.0.1 and manifest release version 1.0.0 disagree/` (strict
   mismatch behavior preserved on the tag-selected file).
7. Worktree strictness: a configured `version_file` that does not exist is
   returned unchanged (no layout probing, no fallback) and the read fails —
   existing strict behavior made explicit.

Synthetic tag fixtures are throwaway `git init`+annotated-tag repositories in
`os.tmpdir()` created by `tests/helpers/tag-repository.ts`; they never touch
refs or tags of the model zoo checkout (no tags created in this repository).
Real-tag cases read only existing published annotated tags.

## 4. Verification runs

- Full `npm run check` (`full-check.log`): rc=0; source validation ok for all
  three worktree platforms; 18 test files, **128 tests passed**; TypeScript
  clean; rebuild reproduces **catalog-v1.0.0-bcb085e3cb057c77** — the current
  worktree artifact identity is byte-identical to the pre-fix build
  (Codex review and `dist/catalog.meta.json` record the same id), so
  reproducibility and asset/hash content are unchanged.
- The exact advertised historical command, post-fix
  (`advertised-pin-after-stdout.log`): rc=0; builds x5 from tag
  `x5-v1.1.2` (`docs/release`), s/x3 from the worktree;
  `catalog-v1.0.0-89ed4880524e4787`, platform_tags
  `{"x5":"x5-v1.1.2","s":"s-v1.1.2","x3":"x3-v1.1.2"}`; provenance in
  `pinned-build-catalog.meta.json` shows x5 kind `tag` with its own manifest
  digest while s/x3 digests equal the current build's
  (`42572ee3…`, `ff4114e1…`). The historical identity is a different catalog
  version than the current one, as expected for pinned content.
- Pinned-build reproducibility (`pinned-reproducibility-check.log`):
  `catalog:check --pin x5=x5-v1.1.2` over the same out directory verifies the
  checksum contract (`catalog-v1.0.0-89ed4880524e4787`).
- Dedicated test run: `tests-run.log`, 128/128.
- Outputs kept out of git: pinned builds under `../.coordination/`
  (outside the worktree), `dist/` is ignored.

## 5. README (`readme.diff`, hash `e0549d28…`)

The pins section now states that a pinned build resolves the platform VERSION
from the layout its tag carries (colocated with the manifest directory, or
platform root in legacy layouts), that the worktree's `version_file` is never
applied to a tag, and that VERSION-less tags and version/manifest disagreement
are rejected. Historical pin instructions, the no-tag-rewrite rule, Node
requirement, npm scripts and the H8-CATALOG-DOC-R1 sections are unchanged.

## 6. Scope

`git status` for `tools/catalog-publisher`: modified `src/sources.ts`,
`tests/sources.test.ts`, `README.md`; new `tests/helpers/tag-repository.ts`.
`sources.json`, `package.json`, `package-lock.json`, workflows, manifests and
`dist/` tracked state untouched. No other task's files
(datasets/Gemma/B10/Codex reports) modified. No installs, downloads, board or
network access; no git commit/push/merge/tag in this repository.
