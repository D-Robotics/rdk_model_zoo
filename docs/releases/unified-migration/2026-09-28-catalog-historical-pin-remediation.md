# H8 (bounded): historical-tag pin VERSION resolution fix — 2026-09-28

Status: **implemented by Claude Code + GLM; round 1 reviewed by Codex with changes-required (version-fallback sequencing); round 2 follow-up implemented — pending independent re-review. H8 is not closed by this document.**

Scope: fix H8-CATALOG-R2 — the advertised
`npm run catalog:build -- --pin x5=x5-v1.1.2 --out …` failed (rc=1) because
`resolvePlatformSources` inherited the current worktree's configured
`version_file` (`docs/release/x5/VERSION`) into the pinned tag read, while the
immutable `x5-v1.1.2` tag keeps its historical root `VERSION`. The fix resolves
VERSION from the layout the selected tag actually carries, keeps worktree
reads strict, and adds regression tests for every tag-layout shape. The
historical pin capability, immutable tags and manifest content are preserved;
no tag, manifest, asset URL/hash, release or site was touched.

## Finding addressed

| Aspect (Codex H8-CATALOG-R2) | Resolution |
| --- | --- |
| Pinned build leaked worktree `entry.version_file` into tag reads | New `resolveVersionFile` in `tools/catalog-publisher/src/sources.ts`: tag sources (pins and `mode: "tag"` entries) resolve VERSION from the selected ref's own layout — colocated `<manifest directory>/VERSION` first, then platform-root `VERSION`, with the pin's tree prefix applied; worktree configuration is never consulted for a tag |
| VERSION must resolve for legacy root, prefixed legacy and unified-layout tags | Probe covers all three: legacy root (`x5-v1.1.2` → root `VERSION`), prefixed legacy (`platforms/x5/VERSION` under the prefix), unified (`docs/release/x5/VERSION` colocated) — each with a dedicated test |
| Genuinely absent VERSION must error cleanly | Shaped error `x5: no VERSION found under <tag> (tried …)`, mirroring the manifest-directory probe's style, instead of a raw `git show` exit 128 |
| Mismatched VERSION must still be rejected | Unchanged downstream `manifestPair` check now runs over the tag-selected file; regression test asserts `/VERSION 0.0.1 and manifest release version 1.0.0 disagree/` |
| Strict current worktree behavior preserved | Worktree sources return the configured path with no probing and no fallback; explicit test proves a missing configured file is returned unchanged and fails the read; full `npm run check` reproduces the identical current artifact `catalog-v1.0.0-bcb085e3cb057c77` |
| Platform source identity, asset URLs/hashes, reproducibility | `PLATFORMS`, `SourceEntry` schema, `sources.json`, emitted `source_ref`/`source_path_prefix`, provenance and asset content untouched; pinned build itself verifies reproducible (`catalog-v1.0.0-89ed4880524e4787`) |

## Verification (host, existing Node v22.23.2 at /opt/homebrew/opt/node@22/bin, existing dependencies)

- Reproduced first: exact advertised command failed rc=1 with the same
  `git show … 128` error Codex recorded, against `src/sources.ts` at HEAD
  `c580b6d9…` — byte-identical to the hash in Codex's
  `historical-pin-failure.json`, so the reproduction matches the reviewed
  state. Evidence: `evidence/…/reproduce-before-{stdout,stderr}.log`.
- After the fix the same command succeeds (rc=0): x5 read from tag
  `x5-v1.1.2` at its real `docs/release` layout with root `VERSION`; s/x3
  remain worktree reads with unchanged digests; result
  `catalog-v1.0.0-89ed4880524e4787` (56 families, 800 benchmarks — historical
  content, distinct identity from the current build as expected).
  `catalog:check` over the pinned output verifies its checksum contract.
- Full `npm run check`: rc=0 — source validation, **128/128 tests**
  (121 + 7 new), TypeScript clean, current-worktree artifact byte-identical
  and reproducible (`catalog-v1.0.0-bcb085e3cb057c77`).
- Known-warnings unchanged (121 x5 / 35 s / 18 x3 accuracy metrics without
  published dataset) — pre-existing source metadata gaps, not synthesized.

## Remaining limits (not claimed)

- No new annotated tags were created in this repository; synthetic tag-layout
  fixtures (prefixed legacy, unified, missing, mismatched) are throwaway git
  repositories under `os.tmpdir()` built by
  `tests/helpers/tag-repository.ts`, fully isolated from the checkout. Their
  temp directories are left for OS cleanup, matching the existing fixture
  test convention.
- No board, toolchain, quantization or download statement is made; catalog
  data content for worktree builds is unchanged (proven by the reproduced
  artifact identity and checksum).
- The pinned historical build's emitted counts/identity describe the frozen
  `x5-v1.1.2` content; they are not migration-completion counts.
- README documents the new pin/VERSION semantics; the data-sources section
  from H8-CATALOG-DOC-R1 is otherwise unchanged.
- H8 remains open (also covers dataset documentation and remaining shared
  integration review); independent Codex review of this package is pending.
- Codex's review report and evidence — including
  `historical-pin-failure.json` — are preserved unmodified
  (sha256 `1092b553…` recorded in `evidence/…/evidence.md`).

## Files

- Modified: `tools/catalog-publisher/src/sources.ts`,
  `tools/catalog-publisher/tests/sources.test.ts`,
  `tools/catalog-publisher/README.md`
- New: `tools/catalog-publisher/tests/helpers/tag-repository.ts`
- New: `docs/releases/unified-migration/2026-09-28-catalog-historical-pin-remediation.md` (this record)
- Evidence: `evidence/2026-09-28-catalog-historical-pin-remediation/`
  (`evidence.md`, before-state captures, diffs, reproduction logs, fixed-run
  logs, full-check log, pinned meta snapshot, pinned reproducibility check,
  test-run log, after-hashes)

## Follow-up round — version-fallback sequencing + README wording (2026-09-28, after Codex version follow-up review)

Status: **implemented by Claude Code + GLM; pending independent re-review. H8 is not closed by this document.**
Scope: `tools/catalog-publisher/src/sources.ts`, `tests/sources.test.ts`,
`README.md`, and this record/evidence set only. Reviewer report and evidence
(`2026-09-28-catalog-version-followup-review.md`,
`evidence/2026-09-28-catalog-independent-review/version-fallback-*`) preserved
unmodified.

### Finding addressed (Codex: resolveManifestDirectory result never reached the VERSION probe)

The round-1 fix pushed `{ ...source, manifestDirectory: await …, versionFile:
await resolveVersionFile(…, source) }` — object-literal fields are evaluated
against the original `source`, so `resolveVersionFile` still saw the
unresolved *preferred* `manifestDirectory` (e.g. `docs/release/x5`) rather
than the probed one. A tag whose manifests resolve through the probe fallback
(`release/models.yaml`) searched `docs/release/x5/VERSION` and root `VERSION`,
ignoring `release/VERSION`:

1. valid `release/VERSION` only → falsely rejected with `no VERSION found`;
2. same plus a stale root `VERSION` → selected the stale root file.

Author reproduction with Codex's preserved two-case `.mts` reproducer (run
from the repository root, unmodified): both cases `correct=false` before the
fix, matching `version-fallback-counterexample.json`.

### Resolution

- `resolvePlatformSources` now resolves the manifest directory **first** and
  passes it explicitly: `const manifestDirectory = await
  resolveManifestDirectory(...)` then `resolveVersionFile(…, source,
  manifestDirectory)`. `resolveVersionFile` takes the resolved directory as a
  parameter (doc comment states the dependency); probe order unchanged
  (colocated `<resolved directory>/VERSION`, then platform-root `VERSION`).
- Worktree strictness and mismatch rejection untouched (worktree mode returns
  the configured path before any directory use).

### Regression tests (+2, suite 128 → 130)

- `resolves VERSION next to a fallback manifest directory`: tag with
  manifests only under `release/` and `release/VERSION` → resolves
  `manifestDirectory "release"`, `versionFile "release/VERSION"` (case 1).
- `prefers the colocated VERSION over a stale root VERSION`: same layout plus
  stale root `VERSION` `0.0.1` → still selects `release/VERSION` (case 2).

### README wording corrections

- "`--pin x5=<annotated-tag>:platforms/x5`" no longer described as "future
  tags containing the new layout": the optional tree prefix is now described
  only for tags whose tree actually carries the platform under a
  `platforms/<id>` prefix.
- "A build therefore always reads the current worktree" qualified: a
  **worktree** build reads the current checkout; a `--pin` build reads the
  pinned tag's tree and its links name that tag.

### Verification (existing Node v22.23.2, existing dependencies, no installs)

- Codex reproducer post-fix (`evidence/…/codex-reproducer-after.log`): both
  cases `correct=true` (`versionFile "release/VERSION"`).
- Full `npm run check`: rc=0 — sources validated, **130/130 tests**
  (18 files), TypeScript clean, current-worktree artifact unchanged and
  reproducible (`catalog-v1.0.0-bcb085e3cb057c77`).
- Exact historical pin preview build: `npm run catalog:build -- --pin
  x5=x5-v1.1.2 --out ../.coordination/catalog-historical-pin-r2` → rc=0,
  identity and bytes identical to round 1 (`catalog-v1.0.0-89ed4880524e4787`,
  sha256 `4778f523…`, 8875279 bytes — the sequencing fix does not alter
  real-tag output); `catalog:check --pin x5=x5-v1.1.2` verifies the pinned
  checksum contract.
- Outputs kept in `../.coordination/` and ignored `dist/`. No board, SSH,
  remote host, download, export, calibration, OE/HMCT or quantization
  activity; no git state changes.

### Round-2 files

- Modified: `tools/catalog-publisher/src/sources.ts`
  (hash `f6a7c44f…`), `tests/sources.test.ts` (`7e05de36…`),
  `README.md` (`5366ecc8…`); helper `tag-repository.ts` unchanged
  (`6107e785…`).
- Evidence additions in `evidence/2026-09-28-catalog-historical-pin-remediation/`:
  `codex-reproducer-after.log`, `full-check-round2.log`,
  `pin-build-round2.log`, `pinned-reproducibility-round2.log`,
  `tests-run-round2.log`, `sources-ts-cumulative.diff`,
  `sources-test-round2-cumulative.diff`, `readme-cumulative.diff`,
  `hashes-after-round2.txt` (also hashes the preserved Codex reproducer and
  review files: `efededee…`, `c9ac27f6…`, `b25da442…`).
- Round-1 evidence files in the same directory are preserved as captured.
