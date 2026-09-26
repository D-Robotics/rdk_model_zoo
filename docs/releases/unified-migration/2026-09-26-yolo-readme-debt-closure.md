# Ultralytics README debt closure — 2026-09-26

Scope: the bounded README baseline originally recorded in B1-R6 is now removed. This does not close B9 sample consolidation, the code-purity audit or the whole host completion plan.

## Completed documentation

The sample root now gives a complete explicit preparation-to-result path, five-task scope, language/target validation boundaries, historical evidence/benchmark entry points, directory responsibilities, source navigation and licensing provenance. It corrects the obsolete X5-only C++ statement: current C++ sources contain both packed-NV12 X5 and split-Y/UV S adaptations, but this round supplies no board validation for them. Historical Python detection evidence is explicitly limited to the recorded checkpoints.

The C++ README now documents the actual task-local CMake projects, repository-root build/run paths, required DNN/UCP/OpenCV SDK layout, detection-only named flags and benchmark, positional-only functional entries, reference-program resource lifecycle and measurement scope. It does not invent a top-level CMakeLists.txt, run.sh, OBB implementation, stable library API or common Python-style CLI. The model README now explicitly names PyYAML as a manifest-reading dependency.

Earlier slices restored model/evaluator/Python/conversion instructions. With the remaining root/C++ debt fixed, the checker passes without exemptions: 36 samples, 0 violations, 37 deliberate rule skips, 0 exemptions. Deleted the baseline JSON and its workflow flag/comment. Checker fixture suite: 27 passed. Generic exemption mechanics remain available for other callers; no rule was weakened.

## C++ verification and fix

The host has no CMake executable. Direct compilation of the exact source sets listed in the existing test CMakeLists first failed because `nv12_geometry.cc` used `std::ptrdiff_t` without `<cstddef>`. Added that standard include. All four executables (`test_decode`, `test_head_probe`, `test_nv12_geometry`, `test_benchmark`) then compiled and passed using the host C++11 compiler. This is host helper verification, not a CMake invocation or board DNN/UCP build.

Five sample-root commands passed host-safe dry-run/list checks in the prepared Python environment; root/C++ bilingual shell command blocks match and all local links resolve. An initial default-shell download-script invocation lacked PyYAML; using the documented dependency environment passed and the dependency is now explicit in model documentation. Board commands were source-reviewed, not executed.

[Structured evidence](evidence/2026-09-26-yolo-readme-debt-closure.json) includes command outputs, compiler/run records and README digests; [original compiler failure](evidence/2026-09-26-yolo-cpp-compile-red.log) shows the portability defect. No board connection, model download, performance measurement or conversion was made.

## Remaining work

Whole-repository root/index/platform README coverage, sample implementation purity, pending B8–B11 migrations, source increments and remaining branch integration stay active. Removing a bounded baseline is not a substitute for source-to-destination capability review or final independent acceptance.
