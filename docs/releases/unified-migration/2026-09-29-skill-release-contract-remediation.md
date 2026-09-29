# Skill release-contract version inventory remediation (SKILL-DOC-R2) — 2026-09-29

Status: **author remediation only, applied by Claude Code + GLM against
SKILL-DOC-R2 in `2026-09-29-skills-behavior-independent-review.md` (reviewer
Codex). This document is not independent acceptance.**

Scope: exactly one paragraph in
`skills/rdk-model-zoo-release/references/release-contract.md` (版本边界,
paragraph after the version-boundary table). No other file was added, edited
or reverted; existing uncommitted changes belonging to other reviewed packages
(including `skills/pack.json`, `skills/README.md`, `skills/CHANGELOG.md`,
`skills/rdk-model-zoo/*`, `skills/tests/*`) were preserved untouched. No
release, tag, workflow, model data, installation, commit, push or merge; no
subagents; no personal memory read.

## Fact correction

The paragraph transcribed a current-member version inventory that had gone
stale: it called the current entry member `rdk-model-zoo` 1.1.1, while the
actual authorities say 1.1.2 (entry advanced 1.1.1 → 1.1.2 for H8-SKILL-R1):

- `skills/pack.json` — Pack `1.1.0`, `release_state: unreleased-candidate`;
  member map: `rdk-model-zoo` 1.1.2, `rdk-model-zoo-repo/develop/validate/review`
  1.1.1, `rdk-model-zoo-integrate/release` 1.0.1.
- `skills/rdk-model-zoo/SKILL.md` frontmatter `version: "1.1.2"`;
  `skills/rdk-model-zoo/skill-card.md` “Skill 版本 | 1.1.2”.

## Change made

The fragile embedded current-member inventory was removed. The paragraph now
states that Pack and member versions are maintained independently and that
release checks must read the actual target-checkout metadata instead of any
transcribed list: Pack version/release state from `skills/VERSION` and
`skills/pack.json` (`version`, `release_state`), history and unpublished status
from `skills/CHANGELOG.md`; each member's version from its own `SKILL.md`
frontmatter `version`, which its governance card `skill-card.md` must match.
These are stated as locations/keys, not as transcribed values, so the text does
not go stale on the next member bump. Paths are inline code spans, not links
(the release skill is independently installed; links out of its root would be
invalid, and `validate_pack.py` enforces local-link closure).

Preserved unchanged: historical upstream facts (upstream `rdk_x5` released
Pack 1.0.1; the old Hub entry `rdk-model-zoo` was 1.0.0 — both facts were
already in the original paragraph and are described as upstream state, this
candidate remaining unpublished) and the independent-versioning rule (Pack and
member versions may differ and are not validated as always equal). The rest of
the file, including the 发布 Skills preconditions paragraph
(`skills/VERSION` 与 Pack 元数据一致), already matched the corrected rule and
was not modified.

## Related version-claim check (exact mismatch only)

Repo-wide search for live claims describing the entry member as 1.1.1:

- `skills/CHANGELOG.md` and `skills/README.md` state the 1.1.1 → 1.1.2
  transition and current levels correctly — untouched.
- `docs/releases/unified-migration/evidence/2026-09-28-skills-increment-remediation/*`
  diffs contain the older sentence as point-in-time evidence — historical
  records, left byte-exact.
- `skills/verification/source-files.sha256` / `source-MANIFEST.sha256` cover
  the edited file but are the prior round's verification snapshot; no live
  test or tool consumes them, and rewriting evidence would falsify it — left
  untouched. They therefore describe that round, not the current file.
- The only live stale claim was `release-contract.md` itself.

## Version governance decision

No member version bump, no new eval scenario: precedent in this pack bumps a
member version for behavior changes (script fix H8-SKILL-R1 → entry 1.1.2) and
upstream increments, while factual doc-consistency corrections (governance-card
version rows, eval-count 70 → 75) were folded in without bumps. The release
skill's instructions/decision flow is unchanged; only a stale fact in a
reference document was corrected to match the already-authoritative metadata.
The skill-card change-review rule (add an eval for new/substantive behavior
changes) is therefore not triggered. Pack stays 1.1.0
`unreleased-candidate`; root VERSION, model manifests and model files
untouched.

## Host check results (2026-09-29, repo `.venv` Python)

| Check | Result |
|---|---|
| `python -m unittest discover -s skills/tests` | 63 tests, OK (includes pack/version-consistency and generator-isolation tests) |
| `python skills/tools/validate_pack.py --pack-root skills` | valid, 7 skills, 84 eval cases, 0 errors; `behavior_evaluated: false` |
| `python skills/tools/sync_references.py --pack-root skills` (read-only drift check) | valid, no drift, nothing applied |
| `git diff` scope | only the one paragraph in `release-contract.md` from this remediation |

Agent behavior, board, SDK, Hub and release verification remain not-run and
are not claimed. Independent review of this remediation belongs to Codex.
