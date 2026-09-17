# P2 — Existing non-equivalent protocols and model composition

> Execute using subagent-driven-development, with independent review and board
> comparisons. The P1 validation record is scoped evidence, not full migration.

## Current delivery scope — user update

The user explicitly narrowed this task to a few representative examples,
without requiring the full Spec to be completed. Finish and deliver ResNet18
classification, DFL/LTRB detection and one PaddleOCR sample with two model stages. Runtime checks for these representatives are recorded, but the user rejected
the full integration claim on 2026-09-17. Source/documentation/conversion
completion is tracked in [the revision plan](2026-09-17-representative-integration.md).
Task 4 below remains an audited future option, not required implementation
for this delivery. P3–P6, seven-skill import and publishing are outside the
current delivery scope; their preparation notes do not imply completion.

## Known source facts

- The canonical `yolo26_det.py` uses direct four-channel LTRB offsets, sigmoid
  class scores, and class-wise NMS. It must not run DFL softmax. Its existing
  `Yolo26Runtime` owns SDK loading and task preprocessing, so the detector does
  not yet accept the P1 callable boundary.
- `yolo26_common.py:ordered_outputs` recognizes the published six tensors by
  NHWC grid shape. Five board baselines from original commit
  `cd74a2b241075bb21036d8d0855d0403f8e8c963` now exist in
  `docs/releases/unified-migration/evidence/*-p2-yolo26-baseline.log`.
- Actual YOLO26n inputs are packed NV12 on X5 and split Y/UV on S100/S100P/S600.
  Outputs are NHWC F32, alternating class channels 80 and box channels 4 at
  strides 8/16/32. The X5 and S compiler names differ. Artifact hashes and
  exact names are in the logs; do not fabricate another asset registry.
- `yolo_v10detect.py` is a different existing case: it keeps DFL but omits NMS.
  Do not equate its postprocessing with YOLO26 based on family naming.
- PaddleOCR is a potential two-model case, not a declared equivalent pipeline.
  Read-only source audit must precede implementation decisions.

## Task 1 — YOLO26 direct-offset detection

- [x] Write a deterministic direct-LTRB fixture that differs from a DFL
  interpretation, wrong-channel/dtype/finite-value tests, and injected runner
  checks. Observe red before implementation.
- [x] Express the direct-offset protocol in a finite local contract. Reuse
  already reviewed NV12 handling and actual image geometry only where semantics
  match. Do not add speculative NCHW/flat/quantized output support.
- [x] Use one image/runner/task orchestration for the truly shared stages, with
  explicit DFL versus LTRB decoding. Keep runtime selection at construction,
  not repeated board-name switches inside every stage. No global base class.
- [x] Preserve original YOLO26 Python class/config names, tuple output,
  scheduling, CLI dispatch, target-dependent ordering, and existing defaults.
  Other YOLO26 task implementations stay unchanged in this task.
- [x] Pass host tests and independent review; compare fixed bus input and exact
  original artifacts on all five boards. Keep comparison tolerances explicit.
- [x] Update contract and bilingual usage/status only after observing results.

Task 1 accepted within its stated scope: [P2 evidence](../../releases/unified-migration/2026-09-16-p2-validation.md),
95 host tests, independent review, and five-board old/new comparison against
runtime snapshot `b109976d4b725cd9fac6191dcd7c08a754bf89bb29a0f970308588b3010da76e`.

## Task 2 — Multi-model source audit and bounded implementation decision

- [x] Record X5/S PaddleOCR source symbols, actual component assets, preprocessing,
  detection geometry, crop generation, recognition/CTC vocabulary, conversion,
  evaluation and old entry paths in `p2-ocr-source-audit.md`.
- [x] Separate demonstrably equivalent behavior from incompatible protocols.
  Form the implementation plan from that evidence, without copying both whole
  pipelines behind a target switch or flattening all inputs into an image API.
- [x] Before changing runtime behavior, obtain exact component metadata and
  original per-target result evidence on available authorized boards.

## Task 3 — Bounded Python OCR composition pilot

The [source audit](../../releases/unified-migration/p2-ocr-source-audit.md)
and the three original-source captures establish an initial implementation
boundary: X5 PP-OCRv3 and S100 PP-OCRv6, each with its own exact detector and
recognizer asset pair. This is a Python composition pilot; old Python/C++ and
conversion paths remain usable and are linked, not declared migrated.

- [x] Create `samples/vision/paddle_ocr/` with one explicit two-stage pipeline,
  detector/recognizer bindings and independently injectable runner callables.
  Runtime loading and scheduling belong in the runner. No SDK or network on
  help/list/dry-run; inference does not implicitly download.
- [x] Bind only the four audited manifest rows and observed input/output
  names, shapes and dtypes. X5 is packed NV12 plus RGB F32 recognition with
  `[1,40,97,1]` output; S100 is split NV12 plus RGB F32 recognition with
  `[1,40,18710]` output. Unsupported targets or mixed pairs fail clearly.
  Use shared manifest resolution and execution-target checks; no new URL table.
- [x] Share normal Python recognition preprocessing and greedy CTC only after
  exact fixtures prove semantics. Keep the fixed X5 alphabet versus checked-in
  S dictionary, detector interpolation, contour edge handling and degenerate
  crop policies explicit in small local policies. Do not copy two entire
  pipelines or import the YOLO/ResNet task implementations as OCR foundations.
- [x] Preserve ordered boxes and texts, including legacy out-of-image boxes.
  Test metadata rejection, output dtype/shape/nonfinite rejection, token count,
  blank/repeat/Unicode decoding, no boxes, rotated crops, repeated invocation,
  and both injected runners. Assert independent lifetime of returned results.
- [x] Add a native CLI with explicit detector/recognizer asset selection and
  local paths, preparation/list/dry-run commands, machine-readable output and
  bilingual usage/status. Keep the S vocabulary source/version visible. Normal
  inference must work from an arbitrary current directory.
- [x] Independently review implementation and host golden comparisons against
  the original source. Then compare stage buffers, boxes/crop pixels and text
  on both X5 boards and S100, using each target's existing default fixture and
  one changed aspect ratio. Record source digest, model/input identity,
  environment, commands and failures. Accuracy/performance remain `not-run`
  until separately measured; S100P/S600 remain unbound without artifact evidence.

Task 3 interface decisions (read-only feasibility review found no structural
blocker):

- Each stage runner is callable with a flat mapping of physical input tensor
  names to NumPy arrays and returns a flat mapping of physical output names to
  arrays. Only the real runtime runner wraps/unwraps the SDK model-name level.
  Stage bindings validate the same tensor contract for real and injected calls.
- The pipeline composes independently replaceable detector and recognizer
  callables. Its result contains ordered, owned boxes and aligned text strings;
  zero detections returns empty collections and never invokes recognition.
  A stage error propagates with stage context; do not silently return a partial
  result or drop a failed crop and break alignment.
- Expose pure prepare/decode operations for stage-level comparison. X5 runtime
  detector data are `[1,960,640,1]` U8 although metadata say `[1,3,640,640]`
  NV12. S100 planes are `[1,640,640,1]` and `[1,320,320,2]` U8. Use the exact
  model/input/output names recorded by the audited metadata captures.
- Raw F32 output names do not prove probability/logit semantics. Preserve
  threshold/argmax behavior without inserting an activation. X5 has 97 classes;
  S100 has 18710 from the original UTF-8 line table, blank at index 0 and the
  appended space token. Preserve file digest
  `769e7fa79bb297b5f18d8dbd149e364a45bc61f2b3f574e5ea836f0b261c23a6`.
- Detection resize is LINEAR on X5 and AREA on S100. Mask resize and ordinary
  recognition resize are LINEAR. Preserve the audited target-local contour
  and degenerate-crop policies. Import SDK and pyclipper lazily.
- Acceptance requires exact U8 input/crop bytes, boxes, text and ordering.
  F32 input buffers must match exactly; raw model outputs initially use
  `atol=1e-6, rtol=1e-5`, with any failure investigated rather than widening
  tolerance silently. This is a same-artifact/board comparison, not a
  cross-platform prediction equivalence claim.

## Task 4 — Existing segmentation and pose contracts

Original-source YOLOv8n segmentation and pose now run on all five authorized
boards. Captures are `evidence/*-p2-seg-pose-baseline.log` under the migration
report directory; remote baseline arrays retain each mask independently and
pose coordinates/scores. Use these real contracts, not inferred YOLO26 outputs.

- [ ] Write the source symbol map and explicit contracts for DFL segmentation
  (three NHWC F32 class/64-channel box/32-channel coefficient groups and one
  `[1,160,160,32]` prototype) and pose (three NHWC F32 class-1/box-64/keypoint-51
  groups). Bind roles by validated metadata, not positional output order.
- [ ] Keep task-specific result types, mask association and keypoint
  association through filtering/NMS. Preserve all existing public config
  defaults, morphology, score policies and mask crop conventions.
- [ ] Move SDK loading to a replaceable runner for these tasks. Reuse only
  demonstrated input/binding/decode helpers, retain separate task modules,
  and do not put all tasks into one universal predict implementation.
- [ ] Test reordered physical tensors, invalid layout/dtype/dimensions,
  empty outputs, non-square image geometry, mask/keypoint alignment, and
  injected callables. Golden tests must compare original source behavior;
  intentional coordinate-policy changes require explicit separate evidence.
- [ ] Review independently, compare original fixed bus results on five boards,
  preserve native and old entry behavior, and record masks and keypoints as
  well as boxes/scores. Do not infer dataset accuracy or performance from this.

Task 4 preflight ruling: the existing shared `post_utils` per-head routines
retain candidate indices through filtering/NMS. The new detection decoder's
three-array return discards that association, so segmentation/pose must not
consume it without an explicit association contract. Existing grids and
coordinates are float64, and old mask cropping depends on integer conversion
of that geometry. Preserve the old coordinate policy and existing common
postprocessing helpers in this task; do not substitute the F32 detector
`inverse_boxes` or its integer-resize correction merely for code reuse. A
separate geometry change would require separate result evidence and disclosure.

Segmentation/pose implementation and later P3 batches remain open; no P2 completion
claim follows solely from adding another detection protocol. No releases,
branch history rewrites, X3 adaptation, or robot control are part of these tasks.
