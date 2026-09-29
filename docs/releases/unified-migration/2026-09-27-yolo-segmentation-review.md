# Ultralytics DFL segmentation — raw stages and mask boundaries

Status: this host increment implemented and verified. Independent Review=not-run,
Board/real SDK/OE/dataset accuracy/latency=not-run; Closed=no.
H2/B9 remain open; this does not close native consolidation or other task audits.
Base: e56fa4f. Sources: S380e1a2; archived source bytes are unchanged.

## Runtime contract and source capability

The standalone S YOLO11 segmentation source dequantizes output_quants before
classification/DFL/coefficient/prototype decoding. The unified wrapper previously
lost that metadata and decoded raw arrays directly. It now uses the same strict
ModelRunner/ModelBinding/RawOutputs boundary as detection. SDK loading, target
identity and descriptors belong to the runner; the task exposes preprocessing,
raw forward, postprocessing and predict/call plus scheduler compatibility.
No model loading, downloading, rendering or file I/O occurs in forward/postprocess.

DFLSegmentationContract describes three NHWC class/64-channel box/32-channel
coefficient heads and stride-4 prototypes. Physical prototype NHWC and NCHW are
supported; affine transforms apply on the physical axis before normalization.
Roles require unambiguous shapes or an explicit complete reviewed name map;
runtime enumeration order cannot silently select a different head. Missing
quantization for integer outputs fails, as do invalid axes, shapes, nonfinite
values and wrong binding carriers through the existing shared validator.
A 32-class ambiguity fixture requires explicit role names to distinguish class
and coefficient tensors with identical shapes.

YoloSeg now returns PreparedDetection with per-image frozen geometry. It preserves
mapping access, forward(prepared), explicit original-width/height postprocessing,
and predict's four-tuple. Plain injected semantic arrays must be floating NHWC;
raw SDK arrays retain their dtype/layout and borrow their original storage until
postprocessing. Returned arrays/masks own storage. The old names-only runtime
fixture was upgraded to full metadata, not bypassed by weakening validation.

## Intentional corrections versus fixed source

- Scalar nonzero zero-point is broadcast for per-channel SCALE, using the earlier
  shared correction. The source helper discards it; do not claim parity for that
  defective case. Independent affine reference fixtures cover the corrected values.
- Geometry uses actual integer resize/padding rather than ideal fractional scale.
- Source prototype crop uses potentially negative slice starts. A RED border case
  reproduced a visible all-one mask becoming empty. Crops now intersect the visible
  image content before slicing, excluding letterbox padding. Fully padded or
  degenerate ROIs yield zero-content masks. No unrelated shared decoder was changed.

Source coefficient/prototype dot-product threshold >0.5, classwise NMS, Lanczos
resize and optional 5x5 opening remain. Output masks are uint8 0/1 box ROI masks,
not full-image masks. This preserves the source math rather than claiming upstream
Ultralytics dataset mask equivalence. Thresholds now fail explicitly on invalid
confidence/NMS values; empty results retain ranks/dtypes.

## Verification and documentation

- Ultralytics 111 tests; shared 144; ResNet 52; OCR 44; checker 27: **378 passed**.
- Migration contract: 44 samples, 0 violations, 46 documented policy skips,
  0 exemptions. No rule or policy downgrade.
- New segmentation tests cover packed/split synthetic inputs, reversed output
  order, prototype layouts, per-channel scales with nonzero scalar offset,
  missing/ambiguous metadata, foreign carriers, invalid inputs/thresholds,
  interleaved geometry, empty/owned results, border and letterbox mask corrections.
- Actual archived S YOLO11 segmentation source plus pinned helper modules execute
  in a host fixture: float, scalar affine and symmetric per-channel cases, two
  resize modes, two geometries and opening on/off (24 combinations) agree for
  interior behavior. Constructors/real SDK are bypassed; no model is executed.
- Runtime README pair contains complete identical segmentation library examples,
  alongside detection examples. Four snippets execute against a fake SDK using the
  actual runner/binder. Link and command evidence is machine captured below.
  README explains output dtype/shape/lifetime, the ROI mask convention, default
  NMS selection, metadata errors and intentional source corrections.

[Evidence and exact commands](evidence/2026-09-27-yolo-segmentation/host-results.json),
[raw results](evidence/2026-09-27-yolo-segmentation/result.json),
[README execution/link evidence](evidence/2026-09-27-yolo-segmentation/readmes.log).
RED logs and the complete final logs are retained alongside the rerunnable host
capture and README verifier. Host fixtures do not prove deployed artifact layout,
SDK ABI, board identity, board accuracy, performance or conversion success.

## Remaining full-scope work

Other Ultralytics tasks (pose, classification, OBB, YOLO26 segmentation, v10),
three standalone native paths, complete iMoonLab conversion/evaluator content,
YOLOE and H0–H9 remain active. Board validation is intentionally deferred by the
user; no board/HP/SSH action was taken. This report is author verification, not
independent acceptance.
