# README quality restoration — host progress, 2026-09-26

Status: in progress. This report is implementation verification, not independent acceptance of the whole migration. The [host completion plan](../../superpowers/plans/2026-09-26-host-completion.md) retains H0–H9 in full. Board execution is excluded by the user's current environment; old board evidence remains valid only for its recorded scope.

## Problems and corrections

The unified Ultralytics model and evaluator READMEs had compressed the source branches' practical instructions into a few short paragraphs. That lost useful preparation, parameter, dataset, output and troubleshooting guidance. This round restores both languages:

- Model: target/format/path table, family/task/scale inventory, defaults and legacy positional form, preview and single/full download commands, offline reuse, checksum limitations, matching inputs/labels and links to conversion/runtime/evaluation.
- Evaluator: COCO/ImageNet/DOTA preparation, all five task commands plus batch evaluation, parameter behavior, output semantics, metric limitations, historical result provenance and common failures. Historical performance is linked directly and is not labeled as remeasured.
- Python runtime: correct the YOLOv8-family/YOLO11-file mismatch, repository-relative launcher/help commands, and the library example's missing image loading/import setup. The full runtime-document review remains pending.
- Fix the full-download wrapper's stale inventory comment (67→92 including YOLO26); no script behavior changed.

## Evidence and boundaries

[Structured evidence](evidence/2026-09-26-yolo-readme-host-check.json) binds the six edited READMEs by SHA-256 and records actual model dry-run output. Seven model command lines ran without downloads; 12 bilingual evaluator examples passed real parser checks. Model/evaluator/runtime code blocks match between languages; all local links resolve. Runtime CLI examples parse, and the API block compiles; it was not executed without a board.

Ultralytics host suite: 78 tests passed. The first run exposed a stale generated catalog (20 URL comparison subcases); rebuilding with Node 22 brought it in line with current manifests, after which all tests passed. This was a generated precondition, not a runtime-code fix. Contract gate: 36 samples, 0 violations, 39 policy skips, 60 exact exemptions. The original 84 exemptions have decreased by 14 evaluator and 10 model findings. No rule was weakened and remaining README debt stays explicit.

No model downloads, dataset evaluations, compiler runs or board inference were performed. Host parser checks cannot prove model availability or numerical/latency accuracy.

## Remaining scope

Review the sample/root READMEs, Python stage contracts and every task's CLI/API, conversion and C++ instructions, then all migrated and pending sample READMEs. Root navigation still needs to reflect the full migration. H0 code/tool fixes and B8–B11 migration remain active; this document does not close them. The remaining Ultralytics baseline and workflow exemption flag must eventually be removed once their actual documentation debt is resolved.
