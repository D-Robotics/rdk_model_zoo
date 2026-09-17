# P2 Task 4 — Segmentation and pose source audit

Date: 2026-09-16. This is a read-only design audit for
[P2 Task 4](../../superpowers/plans/2026-09-16-p2-protocols.md), before its
implementation. The fixed original source is
`cd74a2b241075bb21036d8d0855d0403f8e8c963` on `develop`; the worktree also
contains the completed P1/P2 detection changes. `yolo_seg.py`, `yolo_pose.py`,
and their preprocessing/postprocessing helpers still match that commit after
normalizing line endings. This audit does not cover OCR, YOLO26 segmentation
or pose, C++, conversion, dataset accuracy, or performance.

The smallest supported migration is to share the existing runner, NV12
transport and physical-output validation, while retaining the existing
segmentation and pose postprocessing functions. Both tasks already share
`post_utils.filter_classification`, `decode_boxes`, `gen_anchor`, `NMS`, and
`scale_coords_back`; these functions should be called, not copied. Keep the
task-specific mask and keypoint association, float64 coordinates, legacy
inverse geometry, and crop conventions. The coordinator accepted this
boundary during the audit.

Do not route these tasks through the current three-array `decode_dfl` result
or `geometry.inverse_boxes` merely to increase reuse. The first loses the
candidate indices needed for associated data; the second changes both
coordinate precision and the rounded-letterbox inverse. A small helper that
returns candidate indices could be considered later, with source comparison,
but it is not a prerequisite for this task.

## Evidence and actual artifact boundary

All five saved logs contain two original-source runs, a
`SEG_POSE_BASELINE_PASS` marker, and `EXIT_STATUS=0`. This describes the
coordinator's recorded evidence; the reviewer did not connect to a board.

| Board / saved log | Segmentation instances | Pose instances | Artifact target |
| --- | ---: | ---: | --- |
| [X5 8GB](evidence/x5-8g-p2-seg-pose-baseline.log) | 6 | 4 | X5 / bayes-e |
| [X5 4GB](evidence/x5-4g-p2-seg-pose-baseline.log) | 6 | 4 | X5 / bayes-e |
| [S100](evidence/s100-p2-seg-pose-baseline.log) | 5 | 3 | S / nash-e |
| [S100P](evidence/s100p-p2-seg-pose-baseline.log) | 5 | 4 | S / nash-m |
| [S600](evidence/s600-p2-seg-pose-baseline.log) | 5 | 4 | S / nash-p |

Every record names the same original commit and the same bus image digest,
`c02019c4979c191eb739ddd944445ef408dad5679acab6fd520ef9d434bfbc63`.
The reviewer verified that the local sample image has this digest and BGR
shape `(1080,810,3)`. The capture scripts construct the canonical classes,
use their profile defaults, and set scheduling to priority 0 / core `[0]`.
They save each variable-sized mask separately and save pose coordinates and
scores in remote `baseline-results/yolov8-{seg,pose}.npz` files. The local
logs contain array shapes, dtypes and hashes, not full mask/keypoint arrays.
The reviewer did not retrieve or independently hash those remote files.

The exact qualified manifest references and observed artifact digests are:

| Target / task | Qualified reference | Observed SHA-256 |
| --- | --- | --- |
| X5 / seg | `x5:ultralytics_yolo:yolov8n_seg_bayese_640x640_nv12.bin` | `d8eef15640b93b91a1b8e4182da58db3aef7dc7ba424f701dbf3f711682a2d11` |
| X5 / pose | `x5:ultralytics_yolo:yolov8n_pose_bayese_640x640_nv12.bin` | `01efd851c5fbd898f559ea1310c5c9c942ce2dadc325b05421fa5b5e676c4e60` |
| S100 / seg | `s:ultralytics_yolo:nash-e/yolov8n_seg_nashe_640x640_nv12.hbm` | `34e598f73ccbcd6a557f83d88e4a8992ab5a9b46b354f49c324c76912c11af15` |
| S100 / pose | `s:ultralytics_yolo:nash-e/yolov8n_pose_nashe_640x640_nv12.hbm` | `b2f73777f227baf44017c8817554cb17244f932c5a1a9c9f761fe5418df03e21` |
| S100P / seg | `s:ultralytics_yolo:nash-m/yolov8n_seg_nashm_640x640_nv12.hbm` | `3782d505fb57b73d1c3904ddac4c3d98293a977f0a70df1fcd3f0db2b4ff56e5` |
| S100P / pose | `s:ultralytics_yolo:nash-m/yolov8n_pose_nashm_640x640_nv12.hbm` | `2b1032541eea16544c7d962f5331e61885d2907da1fa00a84aa284833efac54b` |
| S600 / seg | `s:ultralytics_yolo:nash-p/yolov8n_seg_nashp_640x640_nv12.hbm` | `156b09a937c73ee8a718297eec75a73a9acee8cc706769d51ce3b527e26c5fcd` |
| S600 / pose | `s:ultralytics_yolo:nash-p/yolov8n_pose_nashp_640x640_nv12.hbm` | `8d2ea1ebda29c450c3161cfdf0c4cbdc7024b622cd8b68e4341131e5b8e4f32c` |

X5 4GB and 8GB use the same respective artifacts. Publisher SHA-256 is null
in every record; these are observed local digests, not publisher verification.
This table records evidence, not a new asset registry. URLs and filenames
continue to come from the existing manifests.

### Inputs and finite output contracts

All observed models are batch one with logical input `640x640`. X5 reports
`images=[1,3,640,640]`, `hbDNNDataType.NV12`; the existing Python NV12 adapters
send one packed uint8 vector of 614400 bytes, not a three-channel RGB tensor.
S100/S100P/S600 report `images_y=[1,640,640,1]` and
`images_uv=[1,320,320,2]`, both `hbDNNDataType.U8`.

Every output below reports `hbDNNDataType.F32`. The semantic names in this
table are proposed local role labels, not compiler names or implemented APIs.

| Task / semantic role | Physical shape | Meaning supported by source |
| --- | --- | --- |
| Seg `cls_8`, `cls_16`, `cls_32` | `[1,80,80,80]`, `[1,40,40,80]`, `[1,20,20,80]` | 80 raw class logits |
| Seg/Pose `box_8`, `box_16`, `box_32` | `[1,80,80,64]`, `[1,40,40,64]`, `[1,20,20,64]` | Four groups of 16 DFL logits |
| Seg `coeff_8`, `coeff_16`, `coeff_32` | `[1,80,80,32]`, `[1,40,40,32]`, `[1,20,20,32]` | 32 linear mask coefficients |
| Seg `prototype` | `[1,160,160,32]` | NHWC prototype features |
| Pose `cls_8`, `cls_16`, `cls_32` | `[1,80,80,1]`, `[1,40,40,1]`, `[1,20,20,1]` | One raw person-class logit |
| Pose `kpt_8`, `kpt_16`, `kpt_32` | `[1,80,80,51]`, `[1,40,40,51]`, `[1,20,20,51]` | 17 interleaved `(x,y,visibility-logit)` triples |

| Task / target | Stride 8: cls / box / associated tensor | Stride 16 | Stride 32 | Prototype |
| --- | --- | --- | --- | --- |
| Seg X5 | `output0 / output1 / 360` | `368 / 376 / 384` | `392 / 400 / 408` | `419` |
| Seg S100/S100P/S600 | `output0 / output1 / 364` | `372 / 380 / 388` | `396 / 404 / 412` | `423` |
| Pose X5 | `output0 / 344 / 352` | `360 / 368 / 376` | `384 / 392 / 400` | — |
| Pose S100/S100P/S600 | `output0 / 348 / 356` | `364 / 372 / 380` | `388 / 396 / 404` | — |

The reviewer parsed all ten metadata records and verified that these roles
are uniquely identified by the stated shape, with exactly 10 seg or 9 pose
outputs. Enumeration order is not a contract. Prototype and coefficient
tensors share 32 channels but have different grids; both grid and channels
are required to identify them.

New bindings should make these two protocols explicit, validate all roles,
batch, spatial dimensions, dtype and selected configuration at construction,
and validate actual output shape/dtype/finite values on every invocation.
Reject missing, duplicate, extra or ambiguous roles, mismatched grids, wrong
DFL width, wrong prototype/51-channel shape, NCHW/flat metadata, raw integer
or quantized output protocols, and unsupported skeleton/bin counts. The
observed raw dtype is F32; float64 final coordinates are a separate result
property. Do not dequantize hypothetical seg/pose artifacts merely because
the generic DFL detector has a quantization mechanism.

The board evidence establishes YOLOv8n at `640x640`. Existing registry entries
also route YOLO11 seg/pose and YOLOv9 seg to these tasks. Keep their published
entries and validate each artifact against the selected protocol; do not
describe those unmeasured assets, new layouts, or rectangular model inputs as
board-certified by these ten records. Non-square source images are part of
the required Task 4 geometry checks, not evidence for rectangular models.

## Real source and symbol map

Paths in this section are relative to
`samples/vision/ultralytics_yolo/runtime/python/`; line numbers refer to the
audited source snapshot.

| Responsibility | Current symbol / location | Migration boundary |
| --- | --- | --- |
| Seg public config / task | `yolo_seg.py:YoloSegConfig:62`, `YoloSeg:96` | Preserve names and four-item result |
| Pose public config / task | `yolo_pose.py:YoloPoseConfig:69`, `YoloPose:99` | Preserve names and five-item result |
| Load SDK and read input | `YoloSeg.__init__:125`, `YoloPose.__init__:126` → `yolo_runtime.open_model:167` | Move loading to `ModelRunner.from_selection`; construct explicit seg/pose selection |
| Image resize and NV12 | `YoloSeg.pre_process:189`, `YoloPose.pre_process:194` → `preprocess.resized_image:79`, `bgr_to_nv12_planes:42` | Share transport/preparation only with pixel parity |
| Execute model | `YoloSeg.forward:217`, `YoloPose.forward:222` → `model.run` | One replaceable runner invocation, with actual output validation |
| Raw class filtering | Both `post_process` methods → `postprocess.filter_classification:286` | Reuse this existing function and its `valid_indices` |
| DFL offsets / anchors | Both tasks → `postprocess.decode_boxes:411`, `gen_anchor:392` | Reuse these existing functions, preserving numeric precision |
| Mask association | `YoloSeg.post_process:229` → `filter_mces:339` | Same per-head `valid_indices`, concatenation order and NMS `keep` |
| Mask construction | `decode_masks:449` → `resize_masks_to_boxes:657` | Retain task-specific crop, threshold, resize and morphology |
| Keypoint association/decode | `YoloPose.post_process:234` → `decode_kpts:499` | Same per-head indices, anchor and NMS `keep` |
| Suppression | Both tasks → `postprocess.NMS:220` | Reuse existing classwise ordering and IoU comparison |
| Box inverse | Both tasks → `scale_coords_back:167` | Retain legacy inverse for Task 4 |
| Pose inverse / visibility | `scale_keypoints_to_original_image:702`, then `post_utils.sigmoid` | Retain coordinate clipping and exactly one visibility sigmoid |
| Full call / override seam | `YoloSeg.predict:327`, `YoloPose.predict:329`, each `__call__` | Keep separate task methods and dispatch through `self.post_process` |

The public defaults are seg classes 80, mask coefficients 32, morphology on;
pose 17 keypoints; both use resize mode 1, score threshold 0.25, 16 DFL bins,
strides `[8,16,32]`, and derived square anchors unless explicitly supplied.
Canonical `nms_thres=None` becomes the profile value: X5 0.70, S 0.45.
The old canonical constructors require a supplied profile through
`open_model`; the CLI supplies it. Do not confuse this with the separate
YOLO26 constructor's automatic-profile behavior.

## Semantics that must survive the migration

**Candidate association and ordering.** At each stride, classification flattens
NHWC cells in row-major order, takes raw max/argmax, and keeps
`max_logit >= -log(1/threshold - 1)`. That exact index vector selects DFL
rows and either coefficient rows or keypoint triples. Results concatenate in
stride order 8, 16, 32. One classwise NMS `keep` vector must index every
associated array. The helper visits `np.unique(classes)` in ascending order,
uses descending `argsort` scores within a class, and keeps remaining boxes
only when `iou < nms_thres`. Do not independently sort masks/keypoints, add a
new global score sort, or change equality/tie policies during migration.

**DFL precision.** `decode_boxes` uses SciPy softmax over four 16-bin groups,
float32 bin weights, then `gen_anchor`'s float64 `np.linspace` anchors.
Consequently boxes are float64; `decode_kpts` also produces float64 XY.
All five baseline logs confirm float64 boxes/XY, float32 class/keypoint
scores, and int64 IDs. Current `decode.py` normalizes and returns float32
boxes. Even a small rounding change can cross `int()` boundaries used by
mask cropping and change an entire mask shape. Preserve the existing helper
calls rather than tolerating coordinate error while asserting mask equality.

**Segmentation masks.** The current path first takes the prototype's batch
element, then uses `coeff[keep]` and model-space `boxes[keep]`. It scales a
box into prototype coordinates and applies Python `int()` to each boundary.
It slices `protos[y1:y2,x1:x2,:]` before combining coefficients. The binary
test is the raw linear sum **strictly greater than 0.5**, not sigmoid greater
than 0.5. There is no prior clamp of those prototype slice bounds; negative
indices and empty crops follow existing NumPy slicing. This is source
behavior to preserve, not a new recommendation for arbitrary models.

Only afterward are boxes mapped/clipped to the original image. Each cropped
uint8 mask is resized to `max(int(x2)-int(x1),1)` by
`max(int(y2)-int(y1),1)` after the existing box bounds handling, using
`cv2.INTER_LANCZOS4`. Empty masks become zeros of that target shape. With
`do_morph=True`, a 5x5 uint8 ones kernel performs `MORPH_OPEN`. The public
mask result is a list of box-sized arrays, not a dense original-image mask
stack. Do not substitute `process_mask`, full-frame resize, bilinear
thresholding, pre-clamped crops, or a different morphology policy.

**Pose keypoints.** Valid rows reshape to `(-1,17,3)`; XY is
`(raw_xy * 2 + (anchor - 0.5)) * stride`. The third component remains a raw
logit through filtering/NMS and coordinate restoration; the task applies
sigmoid exactly once afterward. Box confidence does not multiply keypoint
confidence. Runtime output does not drop points by the visualization
`--kpt-conf-thres` setting. The task returns `(N,17,2)` XY and `(N,17,1)`
scores aligned with the same boxes and instance order.

**Geometry.** Preprocessing uses nearest-neighbor stretch or linear
letterbox, truncated resized dimensions, integer padding and gray 127.
The legacy inverse deliberately recomputes the ideal uniform letterbox
scale and fractional symmetric padding, then clips boxes/keypoints to
`[0,width]` / `[0,height]`. P1 `inverse_boxes` uses recorded actual X/Y scales
and integer left/top padding and casts to float32. Those are different
policies. The fixed bus image has an exact-ratio resize, so that one board
fixture alone cannot catch rounded-letterbox drift. Preserve the legacy
inverse here; a later geometry change needs its own evidence and declaration.

There is also an existing pure-postprocess compatibility fixture,
`tests/test_runtime_contract.py:13`, for CHW and HWC segmentation prototypes.
`YoloSeg.post_process` transposes a CHW prototype before mask decoding.
The five-board contract is NHWC only. Keep those facts separate: strict
runtime metadata should not gain inferred NCHW support from this fixture,
and withdrawal or relocation of the legacy direct-postprocess behavior must
be explicit rather than silently deleting its test intent.

## Reuse of the completed detection interfaces

| Existing interface | Can share directly? | Concrete limit |
| --- | --- | --- |
| `RuntimeMetadata.from_runtime`, `ModelSelection`, `ModelBinding` | Yes, with explicit task contract | `ModelRunner.from_config` currently creates a detection selection; seg/pose should construct their own selection |
| `ModelRunner.from_selection`, call and scheduling | Yes after binding supports these two protocols | Keep SDK lazy, execution-target check, one load, one call, and surfaced scheduling failures |
| `tensor_io.bind_nv12_inputs` / `InputBinding` | Yes | All ten saved input records bind successfully on the host |
| `tensor_io.OutputBinding.read` | Yes | It can validate extra named roles once complete expected shapes/dtypes are supplied |
| `model_binding._bind_output_roles` | Needs a bounded extension | It currently iterates only cls/box; existing detection contracts require six outputs, not ten/nine |
| `yolo_detect._prepare_image`, `_forward_runner`, scheduling seam | Lower-level behavior is reusable | Keep task results and postprocessing separate; preprocessing parity does not authorize changing inverse geometry |
| Public `decode_dfl` / `_decode_heads` result | No direct replacement | NMS is internal and returned arrays omit per-stride `valid_indices` and final `keep`; boxes become float32 |
| `geometry.inverse_boxes` | No direct replacement in this task | Different rounding, padding and dtype policy |
| Existing `rdk_yolo_utils.postprocess` functions above | Yes, already shared by both tasks | Call them from task code; do not duplicate their algorithms in new task modules |

## Compatibility entry scope

The native entry is `samples/vision/ultralytics_yolo/runtime/python/main.py`
with `--task seg|pose`; `yolo_dispatch.DFL_TASKS/get_task_types` select
`YoloSeg/YoloPose`. `create_runtime_model` passes `--mc` as `mces_num` and
`--nkpt` as `nkpt`, and applies profile defaults. YOLO26 has a different
dispatch table and must remain outside this Task 4 implementation.

| Entry / interface | Behavior that must remain |
| --- | --- |
| X5 `runtime/python/ultralytics_yolo_seg.py` | `UltralyticsYOLOSeg{Config}` names; NMS 0.70; compatibility `mc` overrides canonical `mces_num`; four-item result |
| X5 `runtime/python/ultralytics_yolo_pose.py` | `UltralyticsYOLOPose{Config}` names; NMS 0.70; `classes_num=1` compatibility field; overridden `post_process` returns **three** items: boxes, scores, concatenated `(N,17,3)` keypoints |
| S `runtime/python/yolo_seg.py`, `yolo_pose.py` | Historical names; wrapper defaults NMS **0.7**, anchors `[80,40,20]`, resize 1, score 0.25; detected board profile; canonical four/five-item results |
| X5/S `runtime/python/main.py` | Forward to native CLI; X5 adds `--platform x5` if absent, S resolves target through native behavior |
| Native `evaluator/eval_yolo_seg.py` | Four-item result consumed by `encode_instance_mask`, which pastes each box-sized mask into the full-image COCO RLE |
| Native `evaluator/eval_yolo_pose.py` | Five-item result consumed by `flatten_keypoints`; ordinary DFL visibility remains `1` for score > 0, otherwise `0` |
| X5 legacy evaluators | `eval_Ultralytics_YOLO_Seg_YUV420SP.py` / `eval_Ultralytics_YOLO_Pose_YUV420SP.py` preserve option translation and forward to native evaluators |
| S legacy evaluators | `eval_yolo_seg.py` / `eval_yolo_pose.py` forward to native evaluators |

X5/S entries above are under
`platforms/{x5,s}/samples/vision/ultralytics_yolo/`. The compatibility imports
load canonical task modules under private module names; package imports and
configuration handling must still work in those contexts and from unrelated
working directories. `predict` must continue to honor an overridden
`post_process`, especially the X5 three-item pose adapter.

Do not normalize the S wrapper's 0.7 default to native S's 0.45. Native
evaluator threshold arguments also currently default NMS to 0.70. These
existing entry-specific defaults are distinct from the canonical profile
default, and the saved canonical board runs do not test all of them.

Conversion/mapper assets, C++ pose/segment paths, YOLO26 tasks and historical
benchmark records remain preserved outside this implementation scope.

## Minimal Task 4 implementation and acceptance

1. Add two finite local contracts for DFL segmentation and COCO-17 DFL pose.
   Supply all semantic role descriptors, including prototype/coefficients or
   keypoints, to the existing `OutputBinding`. Extend binding only enough to
   enumerate and validate those reviewed roles; keep six-output DFL/LTRB
   detection behavior unchanged. No new model registry or generic task graph.
2. Let `YoloSeg` and `YoloPose` accept an injected callable and use explicit
   `ModelSelection(task="seg"|"pose", contract=...)` for the real runner.
   Keep loading/scheduling in `ModelRunner`, share NV12 and input preparation,
   and validate all semantic tensors even when confidence filtering is empty.
   Update names-only legacy host fixtures with complete observed metadata;
   do not relax production validation to retain those incomplete fixtures.
3. Replace positional physical-output indexing with role access. Retain the
   existing per-head `post_utils` filtering/DFL decode and all task-specific
   coefficient/keypoint indexing and postprocessing. Preserve separate
   tuple-compatible task results and the legacy override seams. Do not copy
   the helper implementations or refactor them into a universal predictor.
4. Build deterministic original-source golden fixtures for multiple strides,
   overlapping boxes, distinct class IDs, different per-instance coefficients
   and identifiable keypoint triples. Compare IDs/order, mask list length,
   each mask's shape and bytes, XY/scores, defaults and empty outputs. Include
   morphology on/off, exact raw-score thresholds, out-of-image/degenerate
   crops, rounded non-square letterbox, stretch, and repeated calls with
   runtime buffers reused to establish returned-result ownership.
5. Add contract rejection tests for reordered/missing/extra/ambiguous outputs,
   wrong layout/batch/dtype/grid, wrong prototype or keypoint channels,
   nonfinite tensors including filtered-out rows, and injected callables.
   Retain native/old class imports, CLI/evaluator parsing, return formats and
   scheduling behavior. The existing CHW prototype fixture needs an explicit
   compatibility decision alongside the strict NHWC artifact binding.
6. Independently review, then compare final source on the same five boards,
   same respective artifact digests, same bus image and a changed aspect
   ratio. Record the final source digest and exact old/new results, not only
   detection counts. Require exact mask bytes/shapes and IDs/order; set and
   justify coordinate/score tolerances before the comparison. With the old
   math preserved, investigate a mismatch rather than widening tolerance or
   calling it a harmless geometry improvement. Report dataset accuracy and
   performance separately as not-run until actually measured.

## Reviewer checks and remaining risks

This audit ran local source/log inspection and small in-memory host probes,
not a Task 4 implementation test suite:

- Parsed all five logs: ten complete expected metadata records, all original
  source references and fixed input digests consistent, all saved exit markers
  successful. This does not independently rerun those board executions.
- Bound all ten recorded input descriptors with the existing NV12 binding.
  The current six-output detection contract rejected all ten complete
  seg/pose output records, confirming the extension is required.
- Compared old preprocessing with `resize_with_transform` for four source
  shapes `(1080,810)`, `(37,91)`, `(91,37)`, `(100,101)`, each in stretch and
  letterbox mode. All eight resized BGR images and their NV12 planes matched
  element-for-element on the host.
- Confirmed `gen_anchor(8).dtype` is float64. For model box
  `[10,20,40,50]`, source image `37x91`, model `64x64`, legacy inverse Y1 is
  `1.4375000000000049`; the P1 inverse gives `1.423076868057251` and float32
  output. This is a concrete policy difference, not inferred equivalence.
- Confirmed a prototype linear sum of `0.4` yields an all-zero uint8 mask
  under `decode_masks`; adding sigmoid and thresholding at 0.5 would change
  that result.

The highest risks are losing candidate association, changing crop dimensions
through float32 rounding, changing the legacy inverse, silently changing
mask activation/morphology, and breaking X5 pose's three-item adapter. The
current fixed-image logs do not cover score ties, empty results, malformed
metadata, CHW compatibility, buffer reuse or rounded-letterbox edges. Those
are concrete implementation acceptance gaps, not failures of a Task 4
implementation that has not yet been reviewed.

The reviewer made no implementation/test edits, network requests, board
connections, commits, or publications. Only this audit document was written.

## Audited source snapshot

Raw working-file SHA-256; all four source bodies match the fixed original
commit after line-ending normalization:

| File under `samples/vision/ultralytics_yolo/` | SHA-256 |
| --- | --- |
| `runtime/python/yolo_seg.py` | `382a221b74f6ccf0906f0f6c5b313eba4f90884220a2bcdfae915b4e2b11d2ca` |
| `runtime/python/yolo_pose.py` | `b19e67ff5c326295f956ca0e5e764a40b8ca66d4fd8b1cb92921832af77afc28` |
| `runtime/python/rdk_yolo_utils/postprocess.py` | `c9a2477b017ea37aee0faeb3ef0b6cef3b3892dcd83c61f446fa91916bf0cb06` |
| `runtime/python/rdk_yolo_utils/preprocess.py` | `63c2bd0c64fb7d7f397e00d8b631d0437aef8817759f46e5b4633db5b8eb06f4` |
