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

This is an authored delivery candidate, not a published release or a completed source-ownership migration. See the delivery verification report for what was actually tested.
