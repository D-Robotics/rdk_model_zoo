# All Sample readable Runtime implementation plan

> Executor: local Claude Code + GLM. Codex owns dispatch and assessment. Approved design: [全仓设计](../specs/2026-10-05-all-sample-readable-runtime-design.md); accepted exemplars: `samples/vision/resnet/runtime/python/{main,classify,cli}.py` and Ultralytics `detect.py`.

## Acceptance for every batch

- Read repository guidance and sample-specific current classes/README/tests, then record baseline commands and exits.
- Keep edits inside assigned sample directories and assigned release report. Do not modify shared modules, Catalog, root guides or another batch. No staging/commit/push/reset/SSH/submodule initialization.
- Preserve original IO and algorithms. Make main visibly construct the model and call predict; move legitimate CLI helpers to local cli.py, not the full inference flow.
- Put actual stages in the local task class; maintain legacy method/import/constructor compatibility. For shared classification re-exports introduce a local named classifier as in ResNet while retaining old shared-class import for existing callers.
- Add meaningful failing-before/passing-after stage integration checks with injected runtime, then existing regression tests for ALL assigned samples. Include default geometry, custom labels where supported, quantized transforms and sequence/state/multistage failure semantics as applicable.
- Update existing runtime README examples and stage references. Report exact class paths and special semantics. Record logs with source state and not-run boundaries. Stop only for an actual incompatibility, not to ask approval for the accepted architecture.

## Batches

### Batch 1: classifiers (21)

- [ ] Implement and test: `samples/vision/convnext`, `samples/vision/edgenext`, `samples/vision/efficientformer`, `samples/vision/efficientformerv2`, `samples/vision/efficientnet`, `samples/vision/efficientvit`, `samples/vision/fasternet`, `samples/vision/fastvit`, `samples/vision/googlenet`, `samples/vision/hgnetv2`, `samples/vision/mobilenetv1`, `samples/vision/mobilenetv2`, `samples/vision/mobilenetv3`, `samples/vision/mobilenetv4`, `samples/vision/mobileone`, `samples/vision/repghost`, `samples/vision/repvgg`, `samples/vision/repvit`, `samples/vision/resnext`, `samples/vision/vargconvnet`, `samples/vision/vit`.
- [ ] Codex review actual diff and behavioral evidence; fix material regressions; commit only owned paths.

### Batch 2: vision_tasks (13)

- [ ] Implement and test: `samples/vision/3dresnet`, `samples/vision/depth_anything_v2`, `samples/vision/diffusiondrive`, `samples/vision/dinov2`, `samples/vision/fcos`, `samples/vision/lanenet`, `samples/vision/lprnet`, `samples/vision/modnet`, `samples/vision/pointnet`, `samples/vision/pp_liteseg`, `samples/vision/unet`, `samples/vision/unetmobilenet`, `samples/vision/yolo26_depth`.
- [ ] Codex review actual diff and behavioral evidence; fix material regressions; commit only owned paths.

### Batch 3: detection_tracking (5)

- [ ] Implement and test: `samples/vision/ultralytics_yolo`, `samples/vision/yoloe`, `samples/vision/yolov5`, `samples/vision/yoloworld`, `samples/vision/bytetrack`.
- [ ] Codex review actual diff and behavioral evidence; fix material regressions; commit only owned paths.

### Batch 4: multistage (6)

- [ ] Implement and test: `samples/vision/clip`, `samples/vision/siglip`, `samples/vision/efficient_sam`, `samples/vision/mobile_sam`, `samples/vision/paddle_ocr`, `samples/speech/paraformer`.
- [ ] Codex review actual diff and behavioral evidence; fix material regressions; commit only owned paths.

### Batch 5: speech_policy_native (5)

- [ ] Implement and test: `samples/speech/asr`, `samples/speech/kws`, `samples/robotics/himloco`, `samples/llm/gemma4-e2b`, `samples/llm/minicpm5-2b`.
- [ ] Codex review actual diff and behavioral evidence; fix material regressions; commit only owned paths.

### Final integration (51 Samples)

- [ ] Recheck ResNet exemplar, collect a complete coverage matrix: sample, entry, model class/file, stage APIs, actual backend, batch, evidence. Detect missing/duplicate rows against filesystem.
- [ ] Update `docs/architecture/model-examples.md`, new `docs/migration/2026-10-05-all-sample-readable-runtime.md` and repository entry navigation as necessary. Preserve trusted compile/export recipes.
- [ ] Run every native sample test directory in isolated processes, shared tests excluding VLA, sample-contract/metadata/docs/tool tests affected, applicable C++ fixture builds and Catalog build/check. Fix source issues in a fresh bounded Claude task.
- [ ] Run SDK-free entry checks from a clean temporary checkout; dependencies may be reused but record that boundary. No initialized VLA gitlinks.
- [ ] Validate no public legacy imports/CLI flags disappeared without documented compatibility, no `platforms/` resurrected, no remote/download/board results claimed.
- [ ] Codex independently assess evidence, submit complete all-Sample status and remaining board-only checks. Local commits only.
