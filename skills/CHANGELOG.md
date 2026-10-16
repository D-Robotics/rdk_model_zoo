# Model Zoo Skills Changelog

## Pack 1.1.0 — 2026-10-16

Released from the unified `main` delivery branch. Structural validation and
65 host unit tests pass; the 86 Agent evaluation definitions are not executed
behavior results, and no board verification is claimed.

- Content refresh against the unified integration line. Shared references and
  skill entrypoints now state the branch facts — `main` is the delivery default
  branch, `develop` is the unified development/integration line, the preserved
  `rdk_x5`/`rdk_s` branches are historical per-board deliveries, model manifests
  live at `docs/release/{x5,s}/models.yaml` with per-platform version lines
  `docs/release/{x5,s}/VERSION`; a checkout's actual ref is still resolved from
  its Git state, and a cutover that has not happened is never presented as done.
- Branch metadata follows the same split: `pack.json` and `inspect_repo.py`
  record `develop` as the maintenance branch and recognize `main` as the
  unified delivery role (`branch_role: delivery`, layout-gated, no platform
  inference). The integration role is no longer swallowed by the maintenance
  flag, `rdk_x5`/`rdk_s` stay historical per-board targets with their platform
  hints, and detached or unknown refs keep `platform_hint: null`.
- `rdk-model-zoo-release` resolves model versions and manifests from those real
  paths instead of the retired `docs/manifests/` default; the version table
  distinguishes the unified per-platform VERSION files from delivery-line and
  Skills versioning, and names the root `VERSION` as its own unified-repo line.
- `rdk-model-zoo-develop` reads the sample-standards files that actually exist
  (`runtime-code.md`, `readme-contract.md`, `cpp-runtime-code.md`; the retired
  `inference-contract.md` name is gone), states the default single-task layout
  (`main.py` + `cli.py` + one task-named model file; multi-task families keep
  one model file per task), and routes board selection through
  `utils/py_utils/platforms.resolve_target` (`auto|x5|s100|s100p|s600`; unknown
  targets raise).
- `rdk-model-zoo-validate` documents the repository gate split: the
  sample-contract workflow enforces the delivery rule set on every PR,
  host-validation defaults to a single Linux leg with catalog/contract owned by
  their own workflows and an expanded Python 3.10/macOS dispatch matrix, and the
  strict rule set stays a local full-gate opt-in.
- `repository-rules.md` records the current Python surface: `preprocess →
  infer → postprocess` chained by `predict` (old spellings survive only as
  legacy delegates), `main.py` + `cli.py` entries, a new MZ-TARGET-01 check for
  target resolution, and typed detection outputs with explicit dtypes.
- Member versions: `rdk-model-zoo-develop` 1.1.1 → 1.1.2 and
  `rdk-model-zoo-release` 1.0.1 → 1.0.2 for the instruction changes; two eval
  definitions (develop-eval-013, release-eval-011) and the refreshed
  develop-eval-012 cover the retired contract path, the stage naming, and
  Git-resolved branch facts. These changes are included in Pack 1.1.0.
- `rdk-model-zoo/scripts/read_catalog.py` default discovery now also probes
  the unified per-platform layouts `docs/release/{x5,s,x3}/models.yaml` and
  the frozen `platforms/{platform}/...` snapshots, preferring the active
  unified manifest over the same platform's snapshot — consistent with
  `inspect_repo.py`. Discovery stays local to the script (no sibling-skill or
  repository import), so single-skill flat installs remain self-contained.
  Explicit `--manifest` selection and the historic flat single-platform
  layouts are preserved: exactly one candidate is still selected implicitly,
  multiple candidates fail with `ambiguous-manifest` plus the actionable
  `manifest_candidates` path list and are never narrowed by branch, filename,
  or the S group, and missing manifests remain `manifest-not-found`
  (unknown). Implicitly or explicitly selecting a `platforms/` snapshot is
  disclosed with a warning. One eval definition (`zoo-eval-013`, unified
  multi-platform lookup without a selected target) covers this behavior.
- Entry member `rdk-model-zoo` advances 1.1.1 → 1.1.2 for that behavior
  change (pack.json, SKILL.md frontmatter, governance card, pack README,
  structural test expectation).
- OE workspace paths `.drobotics-x5/` and `.drobotics-s/` replace the retired
  `.drobotics/`/`.horizon/` names (upstream `rdk_x5` commit `d1b24f65`), the
  S-series router handoff becomes `drobotics-router`, and the S
  deployment-workflow source link points to `D-Robotics/oe-skills-s` tag
  `v1.1.0` instead of the retired `rdk-skills` path. Upstream declares Pack
  1.0.1 released; that state describes the upstream pack.
- Member versions: entry `rdk-model-zoo` 1.1.0 → 1.1.2 across the two steps
  above; `rdk-model-zoo-repo/develop/validate/review` 1.1.0 → 1.1.1;
  `rdk-model-zoo-integrate/release` 1.0.0 → 1.0.1. The Pack version advances
  1.0.1 → 1.1.0 for the unified delivery content.
- Governance cards restate the real member versions and stay in sync with
  `pack.json`/SKILL.md frontmatter.

## Pack 1.0.0 baseline

- Seven repository-owned Skills with the public `rdk-model-zoo` entry name;
  the review capability is named `rdk-model-zoo-review` (sample audits and
  scoped change reviews).
- The entry Skill reads versioned manifests instead of hardcoded benchmark
  snapshots, separates platform hints from observed compatibility, and
  preserves toolchain handoffs; self-contained context/catalog/evidence
  helpers and explicit workflow boundaries are included.
- Generated, flat-install-safe shared references, governance cards,
  templates, and 84 behavioral evaluation definitions covering unified
  multi-platform catalog lookup, CLI/doc drift, forward boundary violations,
  headings-without-runnable-README, legal OCR multi-stage pipelines, and
  unverifiable board-pass claims.
- `rdk-model-zoo-develop` reads the target ref's sample-standards contracts
  as top applicable clauses and requires file ↔ contract-section ↔ evidence
  mapping before and after each change.
- `rdk-model-zoo-review` applies four mandatory checks — README operability
  walkthrough, inference interface duty boundaries, preservation of prior
  capabilities, and numerical regression evidence — mirrored in the report
  template.
- `rdk-model-zoo-validate` classifies README code blocks as
  explanatory/host/board/conversion before treating any command as verified,
  binds host executions to code SHA, cwd and artifact/input identity, and
  keeps structural validation distinct from command verification.
- `rdk-model-zoo-repo` `inspect_repo.py` reports per-platform manifests under
  `docs/release/{x5,s,x3}` (unified) and `platforms/{platform}/...` (frozen
  snapshots), exposes `platform_manifests` and `unified_layout` facts, and
  reports `branch_role: integration` only when the branch name and layout
  facts agree. Branch names alone never prove hardware; dual-state checkouts
  are flagged.
- Model versions/tags stay separate from the Skills Pack release identity.
