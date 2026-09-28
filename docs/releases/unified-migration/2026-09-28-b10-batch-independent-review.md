# B10 integrated host review — 2026-09-28

Status: **host regression passed; documentation follow-up open**. H6/B10 is not closed. Codex independently reviews; product changes belong to Claude Code + GLM.

## Fresh integrated evidence

`evidence/2026-09-28-b10-batch-independent-review/host-suites.json` records full commands and outputs: ASR 27, KWS 13, Paraformer 46 selected runtime tests, HIMLoco 23, **109 passed**. Paraformer excludes actual conversion/export/calibration/evaluator model execution; model-preparation tests use fixtures. No new board, robot, dataset/model download or quantization execution occurred.

`doc-inventory.json` records hashes of **54 README files**, paired explicit anchor equality and four sample checkers: zero violations, one existing CLI skip per sample, no exemptions. Inline local file links resolve after excluding code blocks; the initial naive scanner's C++ lambda false positives are retained transparently. This scan alone does not prove semantic document quality.

## Semantic audit

Reviewed ASR/KWS model and input guides, conversion availability, saved-prediction scoring schema and metric boundaries against actual scorer code. CTC transcript scoring and KWS confusion/threshold arithmetic are separate from inference; undefined denominators remain null. No fabricated conversion command fills missing source assets. Historical performance remains explicitly historical.

Read HIMLoco conversion/evaluator recipes, input/model descriptions and parameter declarations statically. Current-first 270 observations, raw 12 actions, numerical source IDs, fixed external action scale, independent rollout/reference provenance and source performance scopes remain documented. No real conversion/quantization execution was required or performed.

Read Paraformer root/model/test-data/conversion/evaluator guides against the existing runtime independent review and current frontend/native guides. Three models, explicit CPU CIF, fixed dimensions, prepared-feature handoff and historical versus two-utterance FP32 results are distinguished. Existing export/calibration evidence is preserved as historical work, not rerun as part of this review.

**B10-DOC-R1 remains open:** test-data guides still call the C++ consumer unmigrated. See `2026-09-28-b10-readme-followup-review.md`, already dispatched to Claude Code + GLM. The difference between allowed example output directories merits a clearer prerequisite link, not changing runtime defaults. Until the corrected text is independently reread, these checks do not close H6 or certify the complete batch.

Prior runtime/native reviews remain separate scoped evidence: `2026-09-28-{asr,kws,paraformer,himloco}-independent-review.md`. Actual SDK/board/robot behavior stays not-run. Customer README recipe validation is outside the user-authorized verification scope and is not a blocker.

## B10-DOC-R1 independent follow-up acceptance

Codex reread the four Claude Code + GLM README edits against the Python
application writer, native quickstart/launcher contract and bundled manifest.
The stale native-consumer statement is removed in both languages; the prepared
NPY/manifest handoff and S100 SDK/board not-run boundary are explicit. The
conversion prerequisite now links to the exact producing command and explains
that the three output-directory spellings are examples. The Chinese heading
change is appropriate; it does not claim actual OE/HMCT execution.

Fresh evidence: `evidence/2026-09-28-b10-batch-independent-review/doc-followup-independent-check.json`.
All 17 static checks pass, sample checker has zero violations and one existing
CLI policy skip, and all fenced commands in the four files are byte-identical
to their pre-edit versions. The reviewer inspected the check script before
running it; semantic acceptance is based on the source/doc comparison above,
not headings or string matching alone. No recipe was executed.

**B10-DOC-R1 is closed.** This accepts the documentation correction; H6 final
batch/ledger reconciliation remains separate. The original finding and prior
109-test integrated results above are retained. The author report's native-guide
relative link was corrected by the reviewer as report navigation only.
