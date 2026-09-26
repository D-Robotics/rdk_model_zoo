# Ultralytics detection — raw forward and explicit context

Status: DFL and YOLO26 detection stage changes implemented and host verified.
Independent Review=not-run; Board/real SDK/OE=not-run; Closed=no.
H2/B9 remain open for other tasks, native consolidation and documentation work.
This follows the [source consolidation investigation](2026-09-27-b9-source-consolidation-review.md).

## Corrected runtime boundary

The bound runner previously called OutputBinding.read, which applied affine
quantization in forward. A RED fixture demonstrated integer SDK arrays becoming
float32 before post_process. ModelRunner now calls read_raw_outputs and returns
RawOutputs: an immutable role mapping that retains the actual arrays, their dtype,
layout and binding. No dequantization, activation, layout conversion or detection
filtering occurs in forward. SDK execution still happens exactly once.

Postprocess consumes that carrier and explicitly applies declared transforms.
The carrier distinguishes physical quantized arrays from already-semantic float
maps supplied by injected runners, including when physical names equal role names.
A carrier cannot be reused with a different binding or conceal changed tensor
shapes. Legacy binding.read_outputs remains an explicit postprocess transform;
it is no longer the runner's output adapter operation.

DFL now reads the actual SDK output_quants field as well as existing metadata
aliases, accepts scalar/per-channel SCALE, checks finite positive scales and the
physical axis/channel/zero-point counts, and uses the existing shared affine
implementation. Scalar nonzero zero-points apply to all channels. SDK NONE is
pass-through for floating outputs, not permission to treat integer logits as
floating. LTRB remains floating-only and rejects SCALE. Complex/bool metadata is
rejected even with a descriptor. SDK S8/S16/S32/F16 spellings use shared dtype
normalization. These are declared metadata transformations, not filename guesses.

Raw arrays borrow SDK buffers. Result arrays own their data; a host fixture
reusing all raw buffers after inference leaves the first result unchanged. This
does not prove actual SDK concurrent buffer ownership or thread safety.

## Per-call geometry and compatibility

YoloDetect and YOLO26Detect now return PreparedDetection(tensors, transform).
The frozen transform records actual rounded resize, padding and input geometry;
there is no mutable last_transform or last_image_transform on either task.
Tests prepare different-sized A/B images before postprocessing and verify exact
staged/predict agreement for DFL/LTRB and stretch/letterbox. Input validation
rejects empty, non-uint8 or non-HxWx3 BGR arrays before OpenCV operations.

The task files contain the task/configuration interfaces; common transport and
context helpers moved to detection_io.py. The old tuple preparation adapter lives
in legacy.py. Prepared objects still support [model_name] mapping access, and
forward(prepared) unwraps tensors for old stage callers. Explicit original width
and height can reconstruct geometry without cached state; supplied dimensions
must agree with a supplied transform. Existing predict/call tuple results remain.
The former test reading model.last_transform now checks the returned prepared
transform, alongside the new interleaving tests rather than merely removing the
assertion. SDK execution is not declared thread-safe.

## Source and README checks

Two tests execute actual archived YOLO11 and iMoonLab YOLOv13 postprocessors plus
three archived Python helpers. Their bytes are pinned by SHA-256 to S380e1a2.
YOLO11 scalar affine and symmetric per-channel cases match; YOLOv13 float DFL
matches. These are host tensor fixtures, not board or dataset results. The old
helper's nonzero scalar offset bug under per-channel SCALE is intentionally not
claimed equivalent: the shared correction and independent affine-formula test
cover the intended value instead. The source runtime constructors are bypassed;
no real SDK, model or board is used.

Both runtime READMEs now show the complete three-stage API and compare it with
predict, including all imports and the actual image read. They explain raw-array
lifetime, SDK dtype, metadata errors, context ownership and executable compatibility.
[README verification](evidence/2026-09-27-yolo-stage-purity/readmes.json) checks
118 local links across the sample's READMEs and detection contract, and executes
both identical API examples through the real ModelRunner/binder with a fake S
split-input SDK and mixed float/quantized outputs. Each example calls that SDK
twice, returns one detection and retains raw int8 box tensors.

## Verification and correction of the preceding report

The first complete regression here found the B9 migration-map label was not
parseable once its state became in-progress: its free-text/slashes were read as
a nonexistent sample name. The previous increment's saved final contract log
already had this R-SCOPE error, but its summary manually transcribed rc=0 and
incorrectly generalized an earlier checker27 pass to the final tree. The evidence
and previous report are corrected; original failure logs are retained. The label
now uses the existing parenthetical syntax. No checker rule or exemption changed.

[Final command/return-code record](evidence/2026-09-27-yolo-stage-purity/host-results-final.json)
and complete adjacent logs cover Ultralytics103, shared144, ResNet52, OCR44 and
checker27: **370 tests passed**. Migration scope:44 samples, zero violations,
46 documented policy skips (including the new legacy adapter), zero exemptions.
Initial failed checks and RED tests remain separate from final successful runs.

Open work: DFL segmentation source quantization, pose API semantics, remaining
Ultralytics task-stage interfaces, three native source capabilities, full iMoonLab
conversion/evaluation integration and YOLOE. All other unfinished H0–H9 work is
still required. No new board, HP/SSH, model-body download, real SDK/OE, dataset
accuracy or performance execution occurred. Historical pilot board results refer
to their original commits and cannot certify these changes.
