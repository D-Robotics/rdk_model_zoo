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

## Final non-board batch acceptance

**H6/B10 non-board scope accepted by Codex.** The opening in-progress status
and findings above describe the earlier review stages and are retained.

| Source requirement | Acceptance evidence and current boundary |
| --- | --- |
| ASR Python/C++ chunked audio, explicit decoder policy and resampling behavior | ASR independent review and R1 recheck: 27 tests, corrected integer comparison, CTC/legacy and Python/native resampler differences disclosed; prior five native CTests retain their SDK-double scope |
| KWS source frontend, exact S100 asset and offline scoring | KWS independent review: 13 tests, fixed 60000-sample/373x80 contract, owned raw output and threshold semantics; KWS-N1 source MDTC explanation accepted in the special README review |
| Paraformer multi-model stages, explicit CIF and native WAV-to-feature handoff | Paraformer independent review: 46 runtime tests, six native checks, separate model execution and CPU bridge, empty-token handling and RNG scope; B10-DOC-R1 above closes stale customer handoff prose |
| HIMLoco offline policy, source inputs and raw-action boundary | HIMLoco independent review: 23 tests; 270 observations/12 actions, 21 fixed source observations, external controller scaling, SDK doubles and no actuation |
| Source conversion/evaluation capabilities and historical figures | Original 132-file source audit plus per-sample migration records and this batch's semantic README audit; inherited recipes reviewed as trusted source text, actual quantization execution excluded by user decision |
| Layered bilingual customer documentation | 54-guide inventory and semantic audit above; corrected KWS overview and Paraformer prerequisite/status prose have separate independent acceptance; command blocks and local links checked |
| Integrated host behavior and evidence fidelity | 109 fresh host tests in host-suites.json; final-hash-reconciliation.json compares 176 recorded sample-file hashes and 54 README hashes. Only two KWS root and four Paraformer README changes differ, each independently accepted; runtime evidence has no unexplained drift |

The ledger records done for non-board migration/documentation and host-accepted
for review. Board remains not-run and its overall Closed column remains no,
matching the existing B8 convention. No vendor SDK ABI, model accuracy/latency,
robot control or new quantization result is certified. Whole-repository H1/H8/H9
requirements are not closed by this batch acceptance.
