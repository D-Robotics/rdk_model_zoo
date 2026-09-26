# B8 PP-LiteSeg — host migration and author review

Status: host implementation and documentation complete for this sample;
Board=not-run, independent Review=not-run, Closed=no. This is not B8 acceptance.
Five B8 families remain: UNetMobileNet, YOLO26 Depth, Depth Anything V2, LaneNet
and DiffusionDrive; all wider H0–H9 work remains active.

## Source and behavior

Source: X5 `ac115717197920355fc390bb04299b20e6436864`. One published X5 STDC1
Cityscapes 1024×512 BIN, with unknown publisher SHA-256, retains its exact identity
and URL. No S-series asset or C++ implementation is claimed. Both test images and
the bayes-e YAML are byte-identical to the source. Archived source runtime code
remains unchanged; root README bodies have only a leading canonical-entry banner.

The source runtime expects int32 `(1,512,1024,1)` class IDs, although the old root
and conversion READMEs described logits plus argmax. The canonical implementation
follows the runtime boundary and explicitly rejects logits, wrong geometry/dtype
and IDs outside 0..18. Actual published BIN metadata has not been observed in this
migration: a host fixture is not proof that an external graph or board matches it.

Preprocessing preserves source RGB/NV12 bytes: INTER_LINEAR stretch to 1024×512,
packed NV12 uint8 `(768,1024)`, no CPU normalization. Postprocessing returns owned
int32 `(512,1024)` labels without argmax, dequantization or original-size restoration.
Rendering preserves the source Cityscapes palette and three-panel image exactly.

PPLiteSegTask contains initialization and pre_process/forward/post_process/predict.
Binding, SDK transport/scheduling, CLI file IO and rendering are separate. The runner
reuses SingleArrayRunner rather than duplicating PointNet/UNet loading logic. The
historical no-op scheduling method is replaced with validated runner scheduling.
The evaluator retains --model/--image/--output/--alpha but delegates inference to
the canonical runtime; it adds label/report files and does not claim dataset metrics.
Model preparation is explicit; inference does not download or install dependencies.

## Conversion fixes and limits

Three source failures were reproduced before fixing them:

1. Nested images sharing a basename overwrote a single calibration tensor. The new
   preparation uses unique deterministic filenames, refuses reused output directories
   or manifests, fails unreadable inputs and records source/tensor hashes in a sibling
   manifest. The compiler directory contains only raw tensors. RGB float32 NCHW raw
   0..255 numerics and YAML normalization remain unchanged.
2. Export could proceed without trained weights and install packages implicitly.
   CHECKPOINT must now exist before any external command; config must exist in the
   selected PaddleSeg checkout, and environment installation is a documented step.
3. A successful makertbin exit followed by a missing expected BIN could return zero.
   That branch now exits nonzero; its fake-tool regression is retained.

No checkpoint/PaddleSeg/export dependency version combination is pinned by the
source. Actual graph output compatibility, OE relative path behavior and compiler
artifacts remain unverified. Documentation explains these missing prerequisites
instead of describing the recipe as a proven reproduction of the published BIN.

## README quality

Five levels have ten bilingual READMEs: root, model, runtime/python, conversion and
evaluator. Root includes explicit preparation, supported/not-run matrix, actual
street.png input, output files and no invented street-image result. Model documents
unknown publisher SHA rather than treating an observed digest as authentication.
Runtime lists all parser defaults, fully defined executable stage examples, physical
versus logical input layout, ownership, errors and fixed output geometry.

Conversion retains source export/OE 1.2.8 setup, image/SDK/manual URLs, compiler flags,
normalization, calibration bytes and output paths, while explaining the checkpoint,
version and output-semantic gaps. The source ≈95 FPS/≈10.5 ms expectation and logits
cosine ≥0.95 are retained with their limits: neither is a measured result here, and
class IDs must never be treated as logits for cosine evaluation. Evaluator documents
actual outputs and absence of a dataset loop, mIoU and timing measurements.

All 30 local README links resolve; 11 paired executable blocks match. Both runtime
API snippets execute with the real runner and an injected SDK fixture. Root/sample
indexes include PP-LiteSeg and now cover 39 canonical samples.

## Verification and evidence

- PP-LiteSeg: 18 host tests covering source pre/post/render parity, independent call
  contexts, owned outputs, bad images/class maps/metadata/targets, host-safe CLI,
  real image/NPY/JSON writes, evaluator delegation, README APIs, identity gate,
  manifest downloader selection, calibration bytes/digests/failure cleanup and
  conversion-shell failure behavior.
- Regressions: shared137, ResNet52, OCR44, PointNet21, UNet20, checker27,
  Ultralytics78. Including PP-LiteSeg, 397 tests passed.
- Node22 catalog build/typecheck passed: 57 families, 820 benchmark records.
- Migration contract: 39 samples, 0 violations, 40 CLI policy skips, 0 exemptions.
- Shell syntax checks passed. Initial RED, intermediate runs, final logs, source
  hashes and checker JSON are preserved in [evidence](evidence/2026-09-26-b8-ppliteseg-evidence.json).

No board/HP/SSH action, model download, actual export, OE compilation, dataset
accuracy or real performance test was performed. Independent review remains pending;
these author checks do not close B8 or the overall migration goal.

Whitespace audit reports only trailing blank lines in the byte-preserved source
YAML and raw catalog log; neither evidence/source byte sequence was normalized.
