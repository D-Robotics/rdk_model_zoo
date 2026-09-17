# Representative runtime validation — 2026-09-16 snapshot

Date: 2026-09-16. This report records runtime compatibility checks only.
On 2026-09-17 the user rejected the broader sample-integration completion
claim: source consolidation, README usability and conversion coverage were
insufficient. These numerical results remain valid for their recorded source
snapshots; they do not establish full Sample integration. See the
[revision plan](../../superpowers/plans/2026-09-17-representative-integration.md).

This report separates original-source
baselines from candidate implementation acceptance. P1 evidence remains in
[its own report](2026-09-16-pilot-validation.md), including environment details
and its immutable runtime snapshot. No push, tag, release or deployment has
been performed.

The user subsequently limited delivery to a few representative examples.
The current acceptance scope is ResNet18 classification, DFL/LTRB detection
and the OCR composition pilot. Full P2 segmentation/pose implementation and
P3–P6 are not required for this delivery. Their baseline/audit work below is
preserved as preparation only; it is not a claim of complete Spec delivery.

## Scope and current state

| Task | Original source evidence | Candidate state |
| --- | --- | --- |
| YOLO26n direct LTRB detection | X5 8GB/4GB, S100, S100P and S600; metadata and bus-image result captured | Scoped implementation reviewed; 95 host tests and five-board candidate comparison passed |
| PP-OCRv3 / PP-OCRv6 two-model pipeline | X5 8GB/4GB and S100; metadata, default-image text/boxes, and two-aspect stage buffers captured | Reviewed Python implementation; 31 OCR host tests and three-board, two-aspect stage comparisons passed |
| YOLOv8n segmentation / pose | All five boards; metadata, boxes, scores, IDs, individual masks and keypoint arrays captured | Binding/runner migration not yet implemented |

The fixed original source is commit
`cd74a2b241075bb21036d8d0855d0403f8e8c963`. Source baselines are loaded from an
archive of that commit, not from the developing candidate. Original model
artifacts are selected from existing X5/S release manifests; URLs and observed
digests are retained in per-board logs. Publisher digests are absent for these
assets, so local hashes identify observed bytes without verifying origin.

## YOLO26 baseline

The model has direct four-channel LTRB distances and class-wise NMS. It is not
the NMS-free YOLOv10 protocol, and DFL softmax must not be applied to its four
distances. Outputs are NHWC F32 at strides 8/16/32. X5 and S output names differ.

The X5, S100P and S600 baselines produced five detections with class IDs
`[0,0,0,0,5]`; S100 produced four with IDs `[0,0,0,5]`, using the fixed bus
image, original configuration and explicit scheduling priority/core 0.
The library NMS threshold is 0.45; candidate comparisons use the same
threshold rather than another CLI default.

Evidence (each log has `EXIT_STATUS=0`):

- [X5 8GB](evidence/x5-8g-p2-yolo26-baseline.log)
- [X5 4GB](evidence/x5-4g-p2-yolo26-baseline.log)
- [S100](evidence/s100-p2-yolo26-baseline.log)
- [S100P](evidence/s100p-p2-yolo26-baseline.log)
- [S600](evidence/s600-p2-yolo26-baseline.log)

Candidate acceptance includes boxes (`atol=1e-3, rtol=0`), scores
(`atol=1e-6, rtol=1e-5`), exact IDs/order, native command, returned-result
lifetime and DFL/ResNet regression checks. All five comparisons passed; YOLO26
boxes and scores had **zero maximum absolute difference** from the respective
original baseline, with exact IDs and ordering.

Candidate runtime archive SHA-256:
`b109976d4b725cd9fac6191dcd7c08a754bf89bb29a0f970308588b3010da76e`.
[Per-file hashes](evidence/p2-protocol-source-files.json) identify the tested
snapshot. Report/README edits after this snapshot do not alter runtime code.

| Target | YOLO26 max box / score difference | Candidate evidence |
| --- | --- | --- |
| X5 8GB | 0 / 0 | [comparison](evidence/x5-8g-p2-protocol-comparison.log) |
| X5 4GB | 0 / 0 | [comparison](evidence/x5-4g-p2-protocol-comparison.log) |
| S100 | 0 / 0 | [comparison](evidence/s100-p2-protocol-comparison.log) |
| S100P | 0 / 0 | [comparison](evidence/s100p-p2-protocol-comparison.log) |
| S600 | 0 / 0 | [comparison](evidence/s600-p2-protocol-comparison.log) |

All logs include `P2_PROTOCOL_COMPARISON_PASS` and `EXIT_STATUS=0`. They also
cover runtime compilation on actual board Python, explicit scheduling, a
second inference preserving the first result, native YOLO26 output-image
creation, DFL old/new and legacy-wrapper regression, wrong-target rejection,
and ResNet18 regression on the four boards with a recorded asset (not S100P).

[Host verification](evidence/host-p2-yolo26-tests.log) passed shared 14,
ResNet 29 and YOLO 52 tests (95 total). The independent
[review](2026-09-16-p2-yolo26-review.md) found and closed default-platform
resolution, unsupported LTRB quantization acceptance, and raw-logit class/
threshold ordering defects. These adversarial cases are host evidence;
the board comparison is the fixed-image scope stated above.

## OCR baseline and semantic boundary

The [source audit](p2-ocr-source-audit.md) records the symbol map and real
differences. X5 uses PP-OCRv3 with 97 recognition classes; S100 uses PP-OCRv6
with 18710. They have different artifacts, dictionaries, detector interpolation,
NV12 transport and some contour/crop edge behavior. The new composition must
compare each target with its own legacy source, not equate cross-platform text.

Original source archive SHA-256:
`1848d9686764bb9afaafe598307d981c446464ffddaf45f066c680e1bf7e5bce`.
Default X5 image SHA-256:
`5b4a7fb523c7c459c8d3cec67480c1872cd7b3674b34505467420561ad8c577e`.
Default S100 image SHA-256:
`18a214e1c637fb3a53f71673c6f6a689b5f16d755237ab7e9e58ddc32223580b`.
The additional aspect fixture resizes the same source image to 713x509 using
LINEAR interpolation; this transformation is recorded in each capture script.

| Board | Full legacy result | Per-stage baseline, both aspect ratios |
| --- | --- | --- |
| X5 8GB | [six boxes and text](evidence/x5-8g-p2-ocr-baseline.log) | [capture](evidence/x5-8g-p2-ocr-stages-baseline.log) |
| X5 4GB | [six boxes and text](evidence/x5-4g-p2-ocr-baseline.log) | [capture](evidence/x5-4g-p2-ocr-stages-baseline.log) |
| S100 | [nine boxes and text](evidence/s100-p2-ocr-baseline.log) | [capture](evidence/s100-p2-ocr-stages-baseline.log) |

All these original runs exited zero. Stage captures retain detector input/raw
output, boxes, crop pixels, recognition input/raw output and text. Arrays are
stored in the board's isolated temporary test directory; logs record shapes,
dtypes and per-array byte digests. Corresponding `.py` evidence files contain
the actual capture commands.

Missing `pyclipper==1.4.0` was installed only into the test directory's
`python-deps` on X5 4GB and S100; installed system packages and SDKs were not
changed. X5 8GB already had this dependency. Dependency check/install captures
are preserved next to the baseline logs.

The S manifest has only the S100 OCR pair. S100P fallback and an S600 download
URL in old scripts do not establish binding support. They remain unbound in
the planned pilot. C++ and conversion/evaluation capabilities remain at their
original source paths; this Python pilot does not certify their migration.

The original S100 C++ sample was separately built and run unchanged in the
isolated test directory. [Environment inspection](evidence/s100-p2-ocr-cpp-environment.log)
found CMake 3.22.1, GCC 11.4.0, OpenCV 4.5.4 and the required existing headers;
no system dependency installation was needed. The
[original build/run capture](evidence/s100-p2-ocr-cpp-baseline.log) records
successful configure, build and execution, nine prediction entries and a saved
output image. Its source archive is
`33de0f6e7d2fafd9cccfb887b9b1e9240080681d923cd7912aa72a1006f6623c`;
model, image, vocabulary, font, executable and result hashes are in the log.
This is evidence for the preserved old C++ capability, not a unified C++
implementation or Python/C++ numerical-equivalence claim.

## OCR candidate acceptance

The single PaddleOCR sample composes detector and recognizer stages. Both X5
boards and S100 passed their own original-source comparison for default and
713x509 images. Detector/recognizer input buffers, boxes, crop pixels, text and
ordering match exactly. Maximum absolute raw-output difference was **0** in
all six cases (declared tolerance `atol=1e-6, rtol=1e-5`). Native JSON execution
from another working directory, result ownership across calls, board-side
compilation and mismatched-target rejection also passed.

| Board | Default / changed-aspect text entries | Evidence |
| --- | --- | --- |
| X5 8GB | 6 / 5 | [comparison](evidence/x5-8g-p2-ocr-comparison-attempt1.log) |
| X5 4GB | 6 / 5 | [comparison](evidence/x5-4g-p2-ocr-comparison-attempt1.log) |
| S100 | 9 / 6 | [comparison](evidence/s100-p2-ocr-comparison-attempt1.log) |

All three logs contain `OCR_STAGE_COMPARISON_PASS` and `EXIT_STATUS=0`.
Tested source archive SHA-256:
`c27c8bfeb3148c23b2c1d718913b1acd40e43c8d86ee86004a11a3a4d2d48547`.
[Per-file hashes](evidence/p2-ocr-source-files-attempt1.json) identify the snapshot;
subsequent acceptance-document edits do not change runtime code.

[Combined host checks](evidence/host-p2-ocr-tests.log) passed 14 shared, 29
ResNet, 52 YOLO and the then-current 30 OCR tests. After the final CLI path
collision fix, [all 31 OCR tests](evidence/host-p2-ocr-final-tests.log) passed,
for 126 current tests across those suites. The independent
[OCR review](2026-09-16-p2-ocr-review.md) closed five findings, including metadata
validation, legacy geometry edge behavior and preparation-path collisions.
This establishes fixed-input compatibility, not full-dataset OCR accuracy or
performance. S100P/S600 OCR remain unbound without audited published pairs.

## Segmentation and pose baseline (deferred preparation)

Actual segmentation outputs have three class-80/box-64/coefficient-32 groups
and a `[1,160,160,32]` prototype. Pose has class-1/box-64/keypoint-51 groups.
All observed outputs are NHWC F32. Fixed-image detection counts differ across
artifacts, as expected; candidate checks will compare against each exact
artifact's own result.

| Board | Segmentation instances | Pose instances | Metadata and result evidence |
| --- | ---: | ---: | --- |
| X5 8GB | 6 | 4 | [capture](evidence/x5-8g-p2-seg-pose-baseline.log) |
| X5 4GB | 6 | 4 | [capture](evidence/x5-4g-p2-seg-pose-baseline.log) |
| S100 | 5 | 3 | [capture](evidence/s100-p2-seg-pose-baseline.log) |
| S100P | 5 | 4 | [capture](evidence/s100p-p2-seg-pose-baseline.log) |
| S600 | 5 | 4 | [capture](evidence/s600-p2-seg-pose-baseline.log) |

Every original capture exited zero. Independent masks and keypoint coordinate
and score arrays are retained, not just final renderings. New task contracts,
alignment tests and board comparisons remain pending.

The same old models were additionally run on the bus image resized to 713x509
with LINEAR interpolation, preserving the default fixture results separately.
All five additional captures passed: [X5 8GB](evidence/x5-8g-p2-seg-pose-aspect-baseline.log),
[X5 4GB](evidence/x5-4g-p2-seg-pose-aspect-baseline.log),
[S100](evidence/s100-p2-seg-pose-aspect-baseline.log),
[S100P](evidence/s100p-p2-seg-pose-aspect-baseline.log), and
[S600](evidence/s600-p2-seg-pose-aspect-baseline.log). Their logs include the
transformation, decoded-pixel digest and independent mask/keypoint digests.
The [source audit](p2-seg-pose-source-audit.md) explains why old float64
geometry, mask cropping and candidate association need explicit preservation.

## Unverified scope

Full-dataset accuracy, performance, other model scales, all existing task
protocols, C++ migration, seven-skill import and P3–P6 integration are not
established by these captures. See the [execution plan](../../superpowers/plans/2026-09-16-p2-protocols.md)
and [integration audit](p4-p6-integration-audit.md) for remaining work.
