# Model Zoo Skills Changelog

## Unreleased — candidate Pack 1.1.0

The Pack remains a release candidate (`release_state: unreleased-candidate` in
`pack.json`).

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
  1.0.0 → 1.1.0 as a candidate because the tree now contains 1.1.x members.
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
