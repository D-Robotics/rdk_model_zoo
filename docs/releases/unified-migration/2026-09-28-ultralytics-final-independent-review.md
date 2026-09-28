# Ultralytics runtime composition — independent review in progress

Reviewer: Codex. Baseline 71406056. Status: runtime composition checks accepted;
full H2 documentation/integration closure remains open. No product code was
changed by the reviewer.

## Python responsibilities

Inspected the maintained detection, segmentation, pose, classification and OBB
task classes, YOLO26/S-v10 adapters, detection transport, model runner/binding,
and their runtime API documentation. Tasks compose preprocessing, raw forward
and postprocessing; file reads, rendering and persistence remain in application
modules. Prepared detection geometry travels explicitly with each request.
ModelRunner calls the SDK once and binds raw arrays without activation or decode.
Published maintained YOLO contracts require floating outputs; there is no new
manual integer-dequantization route hidden in forward. Classification applies
softmax/Top-K in its postprocessor. YOLO26 segmentation/pose and S-v10 select
protocols on shared stages; YOLO26 classification aliases the classifier.

The legacy scheduling/call aliases and initialization helpers are compatible
API support, not extra application I/O in inference. Existing YOLO26 detector
grid/map/confidence attributes remain compatibility state; they are not used
by the current decoder. Removal is not required to establish stage separation.
The public raw arrays borrow SDK memory; documentation correctly requires
postprocessing or copying before another inference. SDK concurrency is not
claimed merely because image geometry is per-request.

## Fresh host evidence

[Sample suite](evidence/2026-09-28-ultralytics-final-independent-review/sample-tests.json):
143 tests passed. The captured runtime/test source hashes stayed unchanged
through the checks. Output includes mocked exporter messages and an intentional
export-failure test; no weights or real export/quantization recipe was executed.
A NumPy shape-assignment deprecation warning is preserved.

[Native suite](evidence/2026-09-28-ultralytics-final-independent-review/native-tests.json):
fresh CMake configure/build and all 12 CTests passed, covering six numerical/
geometry/bookkeeping helpers, four X5/UCP output-binding/resource tests and two
production input/task lifecycle tests with ASan/UBSan. These compile against
narrow SDK doubles and do not certify vendor ABI or all board executables.

Native main programs retain the source standalone executable interface and
mature performance paths as permitted by Spec section 5.4; they are not claimed
as a Python-equivalent library API. The C++ README explicitly discloses no OBB
entry, no S-v10 NMS-free coverage, per-task CLI differences and classification
preprocessing differences. Whole-sample closure still requires the in-flight
source-figure documentation package, dataset-navigation update, and final
cross-layer command/catalog review. No board/model/quantization run occurred.

## Final cross-layer command and document check

Codex reviewed model selection, Python CLI defaults, per-task native limitations,
conversion prerequisites and evaluator thresholds/output semantics together.
Runtime help, X5 model listing and five task dry-runs (detect/seg/pose/cls/obb)
pass. Three documented downloader dry-runs also pass with the existing virtualenv
on PATH, including positional syntax and a custom model directory. They only
print asset plans; nothing was downloaded. The first reviewer invocation used
a symlink-resolved system Python directory and failed importing PyYAML. Both the
original errors and corrected runs are retained in `cross-layer-doc-check.json`
and `cross-layer-download-recheck.json` under this review's evidence directory.
This was a reviewer environment error, not a product change.

The 118 recorded runtime/test hashes from the prior 143-test suite have no drift.
Bilingual shell blocks compared after removing comment-only lines are equivalent.
The source figure and dataset navigation packages were separately accepted in
commits 7287b499 and 8cfc4b7b. Native task support is deliberately documented per
program, including absent OBB/S-v10 paths; evaluator metrics, score thresholds
and historical results are not conflated with SDK or board certification.
No real conversion, quantization or board test was run. H2 final requirement
reconciliation and repository-wide H9 remain separate from these scoped checks.
