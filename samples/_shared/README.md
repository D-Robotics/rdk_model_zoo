# Shared sample utilities

These modules serve the migrated samples, including ResNet, Ultralytics YOLO,
PaddleOCR and the two SAM pipelines. They do not define a global model runtime
or a global command.

`platforms.py` reads concrete identity from `docs/release/platforms.json`.
It checks boardinfo first, then X5's socinfo, then the observed device-tree
model. Exact `X5U` was observed on the two test X5 boards; `X5H` and `X5M`
are user-supplied aliases, not additional tested hardware. Explicit selection
is sufficient for preparation; local inference also needs matching identity.

`assets.py` reads the per-platform manifests at `docs/release/{x5,s}/models.yaml`
(moved from the `platforms/` snapshots in Phase 1 A4; the same commit repointed
the reader).
The installed dependency is PyYAML (no Node/publisher or SDK dependency).
Existing manifests have sample IDs plus filenames, without standalone asset
IDs. The new reference `group:sample:filename` qualifies those existing fields;
it does not rename them or claim a new published ID. For example:

```text
x5:resnet:resnet18_224x224_nv12.bin
x5:ultralytics_yolo:yolov8n_detect_bayese_640x640_nv12.bin
s:ultralytics_yolo:nash-p/yolov8n_detect_nashp_640x640_nv12.hbm
```

Selection policy and input/output contracts remain in each Sample. The reader
does not infer S100P support from an `s` data group. URL and expected SHA-256
come directly from the manifest, and absent publisher hashes remain unknown.
`download_asset` is an explicit preparation operation, never called by a task
class. It writes and checks a temporary file before atomic installation and
does not overwrite existing files; existing known hashes are checked too.

Host tests: `python -m unittest discover -s samples/_shared/tests`.
Synthetic data and failure injection do not certify board inference.

## Image bytes

`image.py:bgr_to_nv12_planes` is the single OpenCV I420 → interleaved NV12
implementation used by all three samples. It accepts an even-sized uint8 BGR
image and returns contiguous uint8 `(1,H,W,1)` Y and `(1,H/2,W/2,2)` UV arrays.
The sample adapters preserve their existing validation and error behavior.
Resize/interpolation, packed-versus-split transport and tensor naming remain
sample-local because their contracts differ. NumPy/OpenCV imports are lazy.

The extraction preserves the operation order and chroma byte ordering from
all three implementations. Known-color and noncontiguous-input tests cover
the byte contract; sample tests and affected-board comparisons cover consumers.

## Runtime metadata (Phase 1.5 H3)

`runtime_meta.py:RuntimeMetadata` reads only the public attributes exposed by
`hbm_runtime` objects and never imports a board SDK. A runtime that hosts
several models requires an explicit `model_name` selection — the former
`model_names[0]` assumption is gone — and any number of output tensors is
representable (single-output rules stay with each sample's binding).
`output_quants` carries the per-output quantization descriptors verbatim,
keyed by output name, so task contracts can inspect them without silently
dropping metadata. F32 outputs ignore vestigial quantization descriptors;
the raw values are already floats and must not be dequantized again. ResNet
and PaddleOCR re-export this class as their
`RuntimeMetadata`; each keeps its own contract checks on top.

For evidence writing, `runtime_meta.py:metadata_evidence` projects a
`RuntimeMetadata` (or a plain mapping) into JSON-serialisable values without
copying the SDK descriptors: `dataclasses.asdict` deep-copies every leaf and
the board `QuantParams` type refuses to be pickled, which crashed the
evaluator evidence snapshot on X5 (board evidence 2026-09-24). The projection
keeps names, shapes, dtypes, strides and the complete quant descriptors
(`quant_type`, `scale`, `zero_point`, `axis` plus further public attributes);
unknown objects raise instead of being stringified, and the metadata object is
never mutated. The sample evaluators use it for their `metadata` evidence.

## Declared output transforms (Phase 1.5 H1)

`quantization.py` implements the binding-declared chain from raw runtime
outputs to float32 values: `raw_f32` (float32 passthrough, including vestigial
descriptors; integer dtypes are rejected) and `dequant`
(per-tensor/per-channel SCALE dequantization ported from the delivery
branches' `utils/py_utils/postprocess.py`, rdk_s @ 380e1a2). Activation
semantics are deliberately not part of this chain — whether a raw logit needs
a sigmoid or a dequantized output is already activated is a task-level fact
declared by each sample's `post_process`. The runner validates containers
only; the transform executes in `post_process` with the binding's quant
snapshot.

Affine SCALE decoding uses `(q - zero_point) * scale`. A scalar zero-point
broadcasts to every channel, a vector follows the declared channel axis,
and an empty zero-point means zero. On 2026-09-26 the scalar nonzero case was
corrected: the source helper discarded that offset for per-channel scales.
For example, raw scores `[2, 4]`, scales `[1, 3]`, and zero-point `7` decode to
`[-5, -9]` (class 0), not `[2, 12]` (class 1). This is an intentional source
bug fix, not a new board equivalence claim. Symmetric zero-points, vector
offsets, per-tensor decoding, and non-SCALE passthrough retain their behavior.

## SAM encoder and decoder

[`sam_binding.py`](sam_binding.py) selects exact EfficientSAM/MobileSAM asset
pairs for X5, S100, S100P and S600, then validates both actual models before
execution. Its native metadata snapshot is immutable. [`sam_runner.py`](sam_runner.py)
loads the SDK lazily after hardware identity checks and keeps the source X5/S
container conventions. It never casts or dequantizes outputs.

[`sam_stages.py`](sam_stages.py) exposes encoder and decoder pre/forward/post
methods and an explicit pipeline; [`sam_tensor_io.py`](sam_tensor_io.py) owns
normalization and per-call context. The two samples retain different prompt
protocols, normalization and zero thresholds. Source float32 casts occur in
stage post-processing; no quantization descriptor is silently interpreted as a
dequantization instruction. Result masks stay in the stretched 512-square
image coordinate system.

[`sam_evaluator.py`](sam_evaluator.py) captures legacy/unified stage inputs,
native outputs, results and hashes through the sample evaluator commands. It
requires an explicitly matched board and a new evidence directory in normal
use. Host fixtures test the capture tool; they do not certify model inference.
See the [EfficientSAM guide](../vision/efficient_sam/README.md) and
[MobileSAM guide](../vision/mobile_sam/README.md) for preparation and APIs.

## SCALE descriptor validation

`quantization.validate_scale_quantization` validates positive finite scales,
finite zero-points and per-channel axis/lengths before integer task outputs are
decoded. PointNet and UNet share it; float32 outputs keep their declared raw-float
semantics and are not forced through integer descriptor validation. This helper
checks numeric metadata, not artifact identity or task-specific logits shapes.

## Raw-array runtime transport

`single_array_runner.py:SingleArrayRunner` consolidates the transport shared by
PointNet, UNet, PP-LiteSeg, UNetMobileNet, YOLO26 Depth and Depth Anything V2. Each sample still supplies its selection/binding and physical
input contract: PointNet sends float32 `(1,3,N)` while UNet sends uint8 packed
NV12 `(1,768,512,1)`, distinct from logical model metadata. Local wrapper names and
constructor arguments remain unchanged.

Real loading checks local target identity and artifact integrity before importing
the SDK. Explicit `runtime` / `runtime_factory` injection remains a host test or
caller-provided evaluator seam, not board verification. Metadata binding failure
discards the runtime; scheduling with no arguments remains a no-op. The runner
checks exact tensor names, physical input shape/dtype, output metadata and finite
values, then returns owned raw data. It never performs normalization, dequantization,
activation, argmax, geometry restoration, file IO or model downloads.

The result is one raw array. `physical_input` preserves the one-input API;
`physical_inputs` declares a name-to-shape/dtype mapping for split Y/UV input.
Exactly one contract must be supplied; missing/extra tensors fail before SDK run.
UNetMobileNet uses this split-input form. Multi-stage SAM/OCR,
tracking state and classification-specific contracts are not silently migrated to
it; their behavior needs a separate consumer review before any consolidation.

`single_array_runner.py:NamedArrayRunner` now supplies the same transport for
multiple named raw outputs. LaneNet uses it to retain its float32 embedding,
int64 binary prediction and any observed auxiliary outputs without assuming their
order. The sample binding defines required roles; the transport validates every
observed name, shape, dtype and finite value, then returns owned arrays in a
mapping. `SingleArrayRunner` is a compatible one-output adapter over this base
and still rejects additional outputs. No existing caller changes its return type.
The historical module path remains stable for imports and test injection.

`dequantize_tensor(..., dtype="float64")` is an opt-in precision path for int32
score ordering; the default remains float32 for existing consumers. Explicit
NONE tensors are returned unchanged. This does not infer quantization metadata
or move decoding into the runtime runner.
