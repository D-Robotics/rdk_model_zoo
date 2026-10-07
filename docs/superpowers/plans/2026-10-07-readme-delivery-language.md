# README delivery language and board checklist

Goal: Present the current Model Zoo as a usable deliverable. Preserve every original model preparation and compilation recipe and provide executable board checks for the next session.

Architecture: Keep the accepted 51-sample implementation and the thin main → local model.predict → SDK session structure. README files are user guides; dated release reports hold historical source, migration and verification records. Model implementation stays unchanged. The README-link checker additionally needs a small repair for fenced Python code being mistaken for a link.

Tech stack: Markdown in English and Chinese, pinned Git source, existing sample-contract checks, local Claude Code with GLM 5.3, GitHub Actions.

Spec: User instructions on 2026-10-07 authorize this work, forbid migration/defensive/process-explanation language in README, trust original compilation recipes, and defer real board execution to 2026-10-08.

## Task 1 — Establish source and document inventory

- Read tracked README files outside test fixtures, historical evidence, vendored code and ACT/Pi0 upstream gitlinks.
- Save the baseline source commit and document fingerprints externally. Resolve original X5/S README locations through pinned source and recorded mappings.
- Separate direct technical requirements (SDK, model inputs, actual supported targets) from migration and acceptance commentary.

## Task 2 — Rewrite the user-facing documentation

- Update root README.md / README_cn.md and samples indexes to describe selecting a model, preparing the target environment and running current commands.
- Review all first-party Sample, conversion, model, runtime, evaluator, test-data and facility README pairs. Preserve complete original compilation instructions, arguments, dependency versions, assets, source citations, benchmark values, images and license information; adapt paths to the actual layout. Remove migration-only comparison snippets pointing to the removed platforms tree. Current runtime examples use the local named model and predict.
- Synchronize docs/sample-standards/readme-contract.md and bilingual templates so future documentation follows the same product language. Keep stable anchor IDs and factual support; detailed test status belongs outside README.
- Replace process narration with direct operations or concise technical conditions. Retain actionable restrictions accurately without claiming unperformed validation. Preserve required anchors and full original recipe depth.
- Put any necessary detailed historical acceptance record outside README; do not rewrite past evidence.

## Task 3 — Prepare board checks

- Add docs/validation/board-smoke-test.md. Give current native CLI commands for ResNet and YOLO first, target/artifact/input/output requirements, pass criteria and a result record. Extend using the existing 51-sample index, with native LLM and multi-model/audio requirements explicit. Do not invent a global inference framework.
- Perform no hardware, export, compilation, weights or calibration run.

## Task 4 — Review and integrate

- Independently compare the updated docs with the original recipes and baseline semantic/code/link fingerprints. Inspect every changed paragraph for product wording and bilingual agreement.
- Run the existing contract and relevant documentation tests. Repair real breakage rather than reducing checker scope. The reproduced YOLOv5 false positive requires ignoring fenced code in rendered-link scanning while preserving prose link/image/fragment checks and original line numbers; implement it with failing behavior regressions before the fix.
- Commit reviewed work, integrate develop, and verify CI against the final commit. Keep earlier CI results bound to their own source SHA.

## Execution ownership

The 607 first-party README paths are partitioned without overlap: core/facilities/non-vision/ResNet/YOLO (154), other image classifiers (212), remaining vision (240), and the existing API documentation guide (1). Local Claude Code + GLM authors work only within their assigned paths. A separate read-only Claude task compares the immutable baseline with the original conversion guides. Codex owns the plan, AGENTS.md, documentation contract and templates, independent checks, integration and final CI verification.

The accepted pre-documentation source f773f354 has all three CI workflows passing: host-validation attempt 4 (Linux310 execution4, Linux312 execution3, macOS execution2), sample-contract attempt4, catalog attempt1. Exact artifact digests and full counts were independently verified externally. Final documentation changes require verification against their own final commit.
