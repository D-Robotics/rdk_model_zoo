# UNetMobileNet source audit and shared runtime foundations

This is an intermediate host-work checkpoint, not a completed B8 sample.
UNetMobileNet Python consumer code is still being developed locally; native
C++, full bilingual README coverage and sample integration remain pending.
No board, SDK, model download or conversion execution took place.

## Source audit

All 17 source files match S pin `380e1a2bf42041af54be6f34935e50197cfadff9`.
The [audit](evidence/2026-09-26-b8-unetmobilenet-audit.json) records source hashes,
exact model availability and executable numerical counterexamples.

- Preserve separate S100/S600 HBM assets, INTER_AREA stretch, split NV12,
  original-resolution output and the rdk_colors palette. alpha_f weights the
  original image, unlike PP-LiteSeg's overlay alpha.
- Move visualization out of predict. Source Python returns a blended image;
  the canonical task returns class IDs and the CLI performs rendering.
- Source Python and C++ assume raw int32 argmax always preserves quantized order.
  Counterexample: raw scores [10,2] with scales [1,100] choose class 0 before
  dequantization but class 1 after it. The new implementation must validate
  quantization and decode SCALE only in postprocess, without changing forward.
- C++ adds an intermediate nearest resize to input geometry; Python restores
  directly to original geometry. They can differ for output dimensions that do
  not divide the model input. Canonical behavior follows Python's direct resize;
  actual published output dimensions remain unobserved.
- Native initialization/destruction and task failure paths need resource-safety
  work before acceptance. The source conversion/evaluator directories contain
  only placeholders; no export recipe or dataset benchmark is invented.

## Shared changes delivered by this checkpoint

SingleArrayRunner keeps its single-input API and gains an explicit physical_inputs
name→shape/dtype mapping for split Y/UV. It still returns exactly one owned raw
array. Missing/extra/wrong-type tensors fail before runtime execution. Constructor
requires exactly one physical input contract. SDK loading, identity/file gates,
scheduling and output checks stay shared; no task numerics move into the runner.

Quantization dequantize_tensor gains an optional float64 precision parameter;
existing callers keep the same float32 default. The new comparison path retains
adjacent large int32 scores that float32 can round into a tie. NONE inputs remain
unchanged; this option does not infer quantization or certify metadata.

## Verification

- Shared tests: 139 pass, including split-input transport/rejection/ownership and
  explicit float64 precision versus unchanged default behavior.
- Consumer regressions: PointNet21, UNet20, PP-LiteSeg18, ResNet52,
  Ultralytics78, OCR44, DINOv2 18, FCOS38, YOLOv5 79; total 507 with shared.
- Split-input regression was observed RED before the new constructor contract.
  Full shared's intermediate failure was the not-yet-created UnetMobileNet
  download wrapper in the local work-in-progress tree; adding its real explicit
  preparation entry resolved it without weakening manifest coverage.
- Source numerical counterexamples and raw logs are retained in
  [evidence](evidence/2026-09-26-unetmobilenet-foundations-evidence.json).

Sample ledger remains pending. Continue the complete UNetMobileNet Python/C++
implementation and six-level bilingual docs, then the remaining B8–B11 and H0–H9
checklist. No new independent-review or board acceptance is asserted here.
