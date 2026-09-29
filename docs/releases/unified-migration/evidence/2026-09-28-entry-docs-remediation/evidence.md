# Evidence — 2026-09-28 entry-docs remediation

Author: Claude Code + GLM, working on `codex/b7-board-integration-20260924`.
Scope: `README.md`, `README_cn.md`, `samples/README.md`, `samples/README_cn.md`,
`platforms/README.md`, `platforms/README_cn.md` plus this record and
`docs/releases/unified-migration/2026-09-28-entry-docs-remediation.md` only.

## Contents

- `verification.json` — machine-checked results of the four required
  verifications, run after the edits:
  1. **Fenced blocks unchanged**: every ``` fenced block in all six files
     hashed before and after the edits; all identical (the root README pair
     carries the four blocks; the other four files have none).
  2. **Bilingual inventory 51**: sample-root links parsed from both
     `samples/README*` indexes equal exactly the 51 roots of the reviewer's
     `evidence/2026-09-28-entry-docs-independent-review/inventory.json`
     (45 vision, three speech, one robotics, two LLM; no missing/extra).
  3. **Relative links**: all 496 relative markdown links across the six files
     (56/56/168/168/24/24) resolve to existing files or directories, and every
     fragment anchor (`platforms/x5/README.md#community--contribution`) matches
     a heading in its target. Zero problems.
  4. **Inherited resource links**: each restored URL is byte-identical to its
     occurrence in the archived source guides
     (`platforms/x5/README*.md` at X5 source `ac11571`,
     `platforms/s/README*.md` at S source `380e1a2`).
- `entry-docs.diff` — the complete scoped diff (6 files, +12/−7 lines).

## Fixed sources used

- `docs/releases/unified-migration/2026-09-28-entry-docs-independent-review.md`
  (the review being remediated; its inventory.json is the 51-sample baseline).
- `docs/releases/unified-migration/2026-09-28-minicpm-core-independent-review.md`
  ("Final disposition" section: core package accepted within host scope —
  20 tests, four compiled README examples; B11 as a batch, vendor ABI,
  model precision/performance, board, H8/H9, whole-branch remain open).
- `docs/releases/unified-migration/2026-09-28-b8-batch-independent-review.md`
  (B8/H4 non-board batch accepted; H1/H8/H9 and full migration not closed).
- `docs/releases/unified-migration/2026-09-28-b10-batch-independent-review.md`
  (H6/B10 non-board batch accepted; board not-run; H1/H8/H9 not closed).
- `docs/releases/unified-migration/2026-09-28-b9-batch-independent-review.md`
  (read-only status confirmation: "H5 remains open" — supports the B9/B11
  pending wording; not linked from the product files).
- `tools/catalog-publisher/README.md` "Data sources" (read-only source of the
  corrected R1 description: build resolves from `sources.json`; registry is
  test-only cross-check) and `tools/catalog-publisher/sources.json` (read only).
- Archived guides `platforms/x5/README*.md`, `platforms/s/README*.md` for the
  inherited resource URLs and legacy-archive wording.

## Notes for the reviewer

- Pre-edit file hashes matched the review's `readme_hashes` exactly, so the
  reviewer's inventory applies to the edited baseline.
- No web/network access was used; inherited links are copied verbatim and no
  remote availability or live link status is claimed anywhere.
- The archived X5 guides spell the S legacy org `d-Robotics`
  (`https://github.com/d-Robotics/rdk_model_zoo_s`); the S guides spell it
  `D-Robotics`. The remediation uses the S-guide spelling; recorded here as a
  source-variant note, not reconciled (no claim made about which resolves).
- Platform toolchain-manual URLs were deliberately not restored (they carry
  toolchain-compatibility claim risk the review excluded); the new text points
  to the archived guides for them.
- Other workers' concurrent edits (gemma/bytetrack/yolov5 scopes) are present
  in the shared worktree and are not part of this package.
