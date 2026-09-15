# YOLO26 unified entry implementation plan

Goal: integrate YOLO26 into samples/vision/ultralytics_yolo on local develop.
Spec: user-approved revision of the previous proposal: version selection at one entry, task/output protocols inside it. Do not create a separately maintained YOLO26 Sample.

- [x] Freeze original X5/S YOLO26 code, helpers and catalog for independent comparison.
- [x] Add failing tests for family dispatch, 100 five-task assets, 224 classifier names, output-shape ordering and LTRB decoding distinct from DFL.
- [x] Share platform selection, NV12 binding and existing classification implementation. Integrate YOLO26 LTRB detection, masks, pose and OBB. Preserve documented platform OBB conventions and legacy return adapters. Reject incompatible metadata.
- [x] Add yolo26 family to main/downloader/evaluators; preserve pre-existing family behavior. OBB uses its own task renderer/evaluator. Existing x5 full-download compatibility remains scoped to old families.
- [x] Share five export patches with platform-specific opset/simplify defaults; retain separate X5/S conversion pipelines behind existing mapper entry. No silent installs.
- [x] Forward old YOLO26 CLI, Python classes, converter, exporters and evaluators; retain Benchmark evidence. Keep YOLO26 depth, YOLOE and YOLOv5 out of implementation scope.
- [x] Run synthetic baseline comparisons, new/old CLI and shell tests, existing sample regressions, publisher checks and catalog identity comparison. Record intentional corrections separately from parity results.
- [x] Update bilingual docs, migration report, previous proposal and locally commit. No pushes or claimed board validation.
