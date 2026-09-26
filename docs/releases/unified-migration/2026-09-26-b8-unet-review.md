# B8 UNet — host migration and author review

Status: UNet host implementation complete; independent review not-run,
Closed=no, board not-run. B8 is not complete: six other sample families remain.

## Preserved source capability

Source: X5 `ac115717197920355fc390bb04299b20e6436864`.
All five published ResNet18/34/50/101/152 assets, URLs and publisher SHA-256 values
remain unchanged. The release manifest now points to explicit download.sh.

The full PyTorch architecture with MIT attribution, five bayes-e YAML templates,
mapper guards/logs/receipts, test image, source benchmark tables and PyTorch/ONNX/X5
evaluator are retained. Conversion provenance hashes are in the evidence: the
architecture, mapper and templates remain byte-identical; the exporter changes
only dependency/import handling so host --help needs no torch and imports are
sample-qualified instead of a globally ambiguous `model` module.

The archived platform runtime remains unchanged. Original root README bodies are
preserved with a leading canonical-entry banner. No source C++ existed for UNet,
and none is claimed. S100/S100P/S600 are not supported by these X5 artifacts.

## Runtime and evaluator refactor

- UNetTask contains only initialization and the four business stages. CLI image
  IO/overlay/reporting, VOC coloring, binding and lazy SDK execution are separate.
- Preprocessing preserves the source INTER_LINEAR stretch and packed NV12 bytes.
  A frozen per-call context records original geometry; it is not mutable task state.
- Result remains uint8 512×512 VOC class IDs. It is deliberately not resized back
  to the source image, unlike the different S UNetMobileNet contract.
- Forward preserves raw logits. Integer logits require validated SCALE metadata
  and are decoded only in postprocess; float32 follows shared raw_f32 semantics
  even with a vestigial descriptor. This F32 rule is explicitly documented, not
  claimed numerically equivalent to blindly reapplying a source quant descriptor.
- PointNet and UNet reuse one SCALE descriptor validator, in addition to existing
  shared manifest, identity, metadata and NV12 conversion code. PointNet regression
  tests pass after extraction. Runner consolidation remains part of H8's wider
  shared-responsibility audit; this report does not close H8.
- X5 evaluator now delegates to the canonical task/runner. It still accepts
  explicit newly compiled files, after board/OS checks, and records caller-provided
  artifact status. The normal runtime verifies publisher hashes; evaluation is not
  a backdoor claim that a recompiled model is the publisher's exact file.

## README quality

Five required levels have bilingual guides, and test-data documentation is
preserved. Root includes explicit model preparation, support/verification scope,
results and original reference tables. Model guide retains all five hashes.
Runtime documents every argument/default, executable stage example, logical versus
physical NV12 shape, fixed-size output, scheduling and error handling.

Conversion retains guarded export/mapper commands and adds a complete calibration
preparation example with data/overwrite boundaries. The example was executed on a
copy of the delivered image and checked by the real calibration audit function;
it is a format example, not a representative dataset selection claim.

Evaluator retains all three backends, describes metrics/ignore label/resize scope,
all CLI options, exact report fields and handling of custom compiled filenames.
40 local links resolve and 13 paired command blocks match. Both language API
examples execute under a real runner with an injected SDK fixture.

## Verification

- UNet: 20 host tests, including source preprocessing/postprocessing comparison,
  independent packed NV12 bytes, NCHW/NHWC logits, explicit-stage/predict equality,
  frozen contexts, quantization ownership, malformed inputs, five exact assets,
  board gate before SDK, runner ownership, actual CLI PNG/JSON outputs, evaluator
  delegation/metrics, five YAML templates and executable calibration/API examples.
- Regressions: shared 131, PointNet 21, ResNet 52, Ultralytics 78, OCR 44,
  checker 27; 373 total including UNet.
- Initial implementation RED covered missing modules/CLI. Separate regressions
  reproduced exporter-help's unwanted torch dependency and missing SDK-version
  report field before fixes. An earlier evaluator attempt lacked Pillow; that
  environment failure is not counted as a product regression. Pillow 12.3.0 was
  installed in the existing host venv before the successful evaluator tests.
- Scope: 38 samples / 0 violations / 39 CLI policy skips / 0 exemptions.
- Node 22 catalog build/typecheck passes, 57 families / 820 benchmark records.

[Evidence and logs](evidence/2026-09-26-b8-unet-evidence.json).

## Unverified scope and next work

No board connection, model download, full export/ORT parity run, OpenExplore
compilation or full VOC evaluation was performed. These need actual artifacts,
checkpoints, datasets and toolchains; synthetic SDK metadata is not board evidence.
No independent acceptance is claimed. Continue PP-LiteSeg, UNetMobileNet,
YOLO26 Depth, Depth Anything V2, LaneNet and DiffusionDrive, then the remaining
B9–B11 and full H0–H9 checklist.
