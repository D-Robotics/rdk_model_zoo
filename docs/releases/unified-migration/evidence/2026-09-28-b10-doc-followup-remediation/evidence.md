# Evidence — 2026-09-28 B10 doc follow-up remediation (Paraformer READMEs)

Author: Claude Code + GLM. Docs-only remediation of
`docs/releases/unified-migration/2026-09-28-b10-readme-followup-review.md`
(B10-DOC-R1 + its non-blocking directory-navigation clarity note). Not
independent acceptance. All checks are host-side, static; no download, model,
export/calibration/OE/Mapper/HMCT/quantization, board/SSH/robot access, install,
commit or push was performed.

## Artifacts

| File | Content |
| --- | --- |
| `check_links.py` | Static check script (stdlib only, run from repository root): stale-phrase sweep over every `samples/speech/paraformer/README*.md`; relative-link and explicit-anchor resolution for the four edited files; EN/CN parity (anchor IDs, heading counts, fence counts, byte-identical fenced code blocks) for the two edited pairs; fact checks for each new sentence; `git diff --check` on the edited paths |
| `check_links.out` | Script output: 17/17 checks PASS, zero stale-phrase hits, all links/anchors resolve, both pairs byte-identical in code blocks |
| `sample-check.out` | `tools/sample_contract/check.py --sample samples/speech/paraformer`: 0 violations, 1 skip (pre-existing R-STAGE-PURITY CLI-layer policy skip, same as in `2026-09-28-paraformer-independent-review.md`), 0 exemptions |
| `readme.diff` | `git diff` of exactly the four edited README files at remediation time (4 files changed, +29/−11) |

## Commands (host)

```bash
rdk_model_zoo/.venv/bin/python docs/releases/unified-migration/evidence/2026-09-28-b10-doc-followup-remediation/check_links.py
rdk_model_zoo/.venv/bin/python tools/sample_contract/check.py --sample samples/speech/paraformer
rdk_model_zoo/.venv/bin/python -m unittest discover -s tools/sample_contract/tests   # 27 tests OK
```

## Files changed by this remediation

- `samples/speech/paraformer/test_data/README.md` — B10-DOC-R1: stale "still being
  migrated" replaced by implemented prepared-feature handoff + native-guide link +
  explicit host-checked-only / S100 SDK/board not-run boundary.
- `samples/speech/paraformer/test_data/README_cn.md` — Chinese mirror of the same.
- `samples/speech/paraformer/conversion/README.md` — clarity: which documented
  preparation produces the `--feature` example path; example-directory naming
  explained; no command, flag or default changed.
- `samples/speech/paraformer/conversion/README_cn.md` — Chinese mirror + one stale
  heading (`源流程与待迁移部分` → `源流程与剩余迁移边界`, left over from commit
  `6a7add6b` before the conversion stages were implemented; flagged in the report
  as an author judgment call for independent confirmation).
