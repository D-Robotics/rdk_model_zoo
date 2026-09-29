# H8-SKILL-R1 read_catalog unified-layout discovery — author record

Base: `d2d2a4e0` working tree (the shared-navigation README correction is still
uncommitted in the same tree and is not re-reviewed here). Implementation by
Claude Code + GLM; Codex reviews and owns Git synchronization. Board runs,
real SDK, downloads, quantization, and whole-H8 closure are out of scope and
remain not-run. Design and file map below were fixed before implementation;
verification sections are filled from actual runs.

## Problem

Finding H8-SKILL-R1
([2026-09-28-skill-catalog-layout-review.md](2026-09-28-skill-catalog-layout-review.md)):
the documented default invocation
`read_catalog.py --repo <unified checkout> --model efficientnet` returned
rc=2 `manifest-not-found` although the active unified manifests
`docs/release/x5/models.yaml` and `docs/release/s/models.yaml` exist. The
script only probed flat single-platform layouts (`docs/manifests/`,
`docs/release/`, root `release/`), so the unified checkout produced a false
absence instead of requiring target selection. `inspect_repo.py` already
discovers all three present manifests (both active plus the historical
`platforms/x3/release/` snapshot). Reproduced before any change, with the
explicit-selection success and the `inspect_repo` cross-check, in
`evidence/2026-09-28-skill-catalog-layout-remediation/before-real-checkout.json`;
the original Codex command evidence stays in
`evidence/2026-09-28-shared-navigation-independent-review/tools-and-context.json`.

## Design (recorded before implementation)

Discovery lives entirely in `read_catalog.py`; nothing is imported from a
sibling skill or from repository-only shared helpers, so a single copied
`rdk-model-zoo/` directory stays self-contained.

1. **Explicit selection first** — a passed `--manifest` is used unchanged
   (existing safe-path/traversal rules and YAML limits untouched).
2. **Candidate set** — flat layouts (`docs/manifests/`, `docs/release/`,
   root `release/`) plus known per-platform layouts probed in preference
   order `docs/release/{x5,s,x3}/` (unified, active) →
   `platforms/{platform}/docs/release/` → `platforms/{platform}/release/`
   (both frozen migration-window snapshots). The first present pattern wins
   per platform, so an active unified manifest shadows the snapshot copy of
   the same platform — consistent with `inspect_repo.py`'s
   `PLATFORM_MANIFEST_PATTERNS` order. Candidates are sorted for deterministic
   reporting; branch names, filenames, and the S release group are never
   consulted.
3. **One candidate → implicit selection** — preserved historic behavior for
   single-platform refs (flat and now also a single per-platform manifest).
4. **Multiple candidates → explicit requirement** — fail with
   `ambiguous-manifest` carrying the actionable candidate paths both in
   `reason` and in a new structured `manifest_candidates` field. On this
   checkout the default invocation therefore reports three candidates
   (`docs/release/s`, `docs/release/x5`, `platforms/x3/release`) instead of a
   false absence.
5. **No candidates → `manifest-not-found`** — unchanged reason string;
   missing manifests stay unknown and are never replaced by maintenance-source
   data.
6. **Snapshot disclosure** — selecting (implicitly or explicitly) a
   `platforms/` manifest adds a warning that it is a frozen migration-window
   snapshot, not the active unified release location. Active/snapshot data is
   otherwise read identically; no metrics are invented either way.
7. **Governance** — entry member `rdk-model-zoo` advances 1.1.1 → 1.1.2
   (pack.json, SKILL.md frontmatter, governance card, pack README, structural
   test). Pack stays candidate 1.1.0 with `release_state: unreleased-candidate`;
   nothing is published, tagged, installed, or synced to Hub. Per the
   governance card's change-review rule one eval definition (`zoo-eval-013`)
   is added; it is a definition, not an executed result.

## File map

| File | Change |
|---|---|
| `skills/rdk-model-zoo/scripts/read_catalog.py` | discovery constants + `manifest_candidates()`, `AmbiguousManifest`, snapshot warning, structured candidates on failure, discovery help text |
| `skills/tests/test_tools.py` | six catalog regressions (below) + `shutil` import |
| `skills/tests/test_pack.py` | entry version expectation 1.1.2; README restatement assertions |
| `skills/README.md` | entry version line, `read_catalog` discovery paragraph, eval count 83 → 84 |
| `skills/rdk-model-zoo/SKILL.md` | frontmatter 1.1.2; instruction step 2 describes candidate reporting and the no-inference rule |
| `skills/rdk-model-zoo/skill-card.md` | `Skill 版本 1.1.2` |
| `skills/rdk-model-zoo/evals/tasks.yaml` | `zoo-eval-013` (unified multi-platform lookup without a selected target) |
| `skills/CHANGELOG.md` | Unreleased bullets for the fix, the version bump, and the review finding |
| `skills/pack.json` | `rdk-model-zoo` 1.1.2 |

`skills/README.md` also still carries the accepted-but-uncommitted
shared-navigation correction; this change only rewrites the `read_catalog`
sentences inside that paragraph and does not undo it.

## New regressions (all RED before the fix, GREEN after)

`evidence/2026-09-28-skill-catalog-layout-remediation/red-new-tests-before-fix.log`
records the six failures against the pre-fix script; all 15 pre-existing
catalog tests passed unchanged in the same run:

- `test_unified_multi_platform_requires_target_selection` — x5 + s unified and
  the x3 snapshot present: rc=2, `ambiguous` + `--manifest` in the reason,
  exact three-path `manifest_candidates`; a `--model` query must not narrow
  the target either.
- `test_single_per_platform_manifest_is_implicit` — only
  `docs/release/x5/` present: implicit selection with the companion
  `benchmarks.yaml` resolved from the same directory.
- `test_unified_manifest_wins_over_same_platform_snapshot` — distinct model
  ids prove the unified content was read, no snapshot warning.
- `test_snapshot_only_selection_is_disclosed` — snapshot-only checkout
  selects it implicitly and warns.
- `test_mixed_flat_and_per_platform_requires_selection` — flat
  `docs/manifests/` next to unified `docs/release/x5/`: ambiguous, both
  candidates listed (historic flat layout is not silently preferred).
- `test_script_runs_without_sibling_skills` — the script copied alone into an
  empty directory with `PYTHONPATH=` still discovers and reads a per-platform
  manifest (self-containment guard).

Existing coverage retained and passing: explicit two/three-flat ambiguity,
missing-manifest `manifest-not-found` (exact reason), symlink-escape and
cyclic-YAML rejection, unsafe-tag rejection, duplicate ids, redefined YAML
anchors, S-group scope warning, null/manual preservation.

## Verification (actual runs)

- Pre-fix real-checkout reproduction: default rc=2 `manifest-not-found`;
  explicit `docs/release/s/models.yaml` rc=0; `inspect_repo` lists all three
  manifests (`before-real-checkout.json`).
- Post-fix real checkout (`after-and-host-checks.json`): default invocation
  rc=2 `ambiguous-manifest` with the three candidate paths; explicit
  `--manifest` succeeds for `docs/release/x5`, `docs/release/s`, and the
  historical `platforms/x3/release` snapshot (the x3 selection carries the
  snapshot warning and reports zero `efficientnet` records without
  fabricating any).
- `unittest discover -s skills/tests`: 63 tests OK (57 before, +6).
  `validate_pack.py`: valid, 7 skills, 84 eval cases. `sync_references.py`
  read-only: valid, no drift, nothing written. `git diff --check`: clean.
- Python: `../rdk_model_zoo/.venv/bin/python` (3.13.x); no packages installed.

## Out of scope / not-run

Agent behavior evaluations (zoo-eval-013 is a definition only), Hub
installation or publishing, model manifests' content (URLs/checksums/benchmarks
untouched), other skills' scripts, board or SDK execution, and whole-H8
closure. Codex independently reviews this remediation; only it syncs Git
state.
