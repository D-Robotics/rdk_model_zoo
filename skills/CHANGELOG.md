# Model Zoo Skills Changelog

## Unreleased — candidate Pack 1.0.0

- Maintain one cross-platform source on `rdk_x5`; determine each task's target platform and revision independently.
- Support current `docs/manifests/` alongside historical manifest layouts, with explicit ambiguity handling.
- Re-run candidate checks on the maintained repository and record local verification separately from Agent and hardware evidence.

- Add seven repository-owned Skills and preserve the public `rdk-model-zoo` entry name.
- Rename the review capability to `rdk-model-zoo-review`; support sample audits and scoped change reviews.
- Refactor the migrated entry to Skill 1.1.0: replace hardcoded benchmark snapshots with versioned manifest reading, separate platform hints from observed compatibility, and preserve toolchain handoffs.
- Add self-contained context/catalog/evidence helpers and explicit workflow boundaries.
- Add generated, flat-install-safe shared references, governance cards, templates, and 75 behavioral evaluation definitions (70 original and five cross-platform additions).
- Keep model versions/tags separate from the proposed Skills Pack release identity.
- Execute the 75-case Codex core suite and seven paired controls, preserving failures, incomplete setup and one control timeout; record focused follow-up runs separately.
- Clarify repository inventory versus code review discovery after observed misrouting; retain explicit PR and staged-review entrypoints.
- Correct evaluation fixtures and scope-conflicting assertions with frozen before/after evidence; do not equate host checks or synthetic model records with hardware acceptance.

- Strengthen `rdk-model-zoo-develop` 1.1.0: read the target ref's sample-standards contracts (readme-contract, inference-contract) as top applicable clauses when present, and require file ↔ contract-section ↔ evidence mapping before and after each change instead of a single "docs updated" line.
- Strengthen `rdk-model-zoo-review` 1.1.0 with four mandatory checks — README operability walkthrough (complete headings do not statically pass), inference interface duty boundaries (download/NMS/decoding/drawing/file output inside forward is a finding), preservation of prior capabilities, and numerical regression evidence — and mirror them in the report template.
- Strengthen `rdk-model-zoo-validate` 1.1.0: classify README code blocks as explanatory/host/board/conversion before treating any command as verified, bind host executions to code SHA, cwd and artifact/input identity, and keep structural validation distinct from command verification.
- Extend `rdk-model-zoo-repo` 1.1.0 `inspect_repo.py` for the unified migration layout: report per-platform manifests under `docs/release/{x5,s,x3}` (unified) and `platforms/{platform}/...` (frozen snapshots), expose `platform_manifests` and `unified_layout` facts, and report `branch_role: integration` only when the branch name and layout facts agree. Branch names alone never prove hardware; dual-state checkouts are flagged.
- Add eight behavior-evaluation definitions (develop 2, review 4, validate 2; totals 12/14/12) covering CLI/doc drift, forward boundary violations, headings-without-runnable-README, legal OCR multi-stage pipelines, and unverifiable board-pass claims. These are definitions of expected behavior, not executed results; no behavior-passing claim is implied.
- Shared content was not modified, so `references/` copies are unchanged by `sync_references.py`; pack version stays candidate 1.0.0 while member versions advance independently.

This is an authored delivery candidate, not a published release or a completed source-ownership migration. See the delivery verification report for what was actually tested.
