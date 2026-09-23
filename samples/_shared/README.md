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
keyed by output name, so F32 contracts can reject them instead of silently
dropping metadata. ResNet and PaddleOCR re-export this class as their
`RuntimeMetadata`; each keeps its own contract checks on top.

## Declared output transforms (Phase 1.5 H1)

`quantization.py` implements the binding-declared chain from raw runtime
outputs to float32 values: `raw_f32` (passthrough; integer dtypes or reported
quantization descriptors are contract mismatches) and `dequant`
(per-tensor/per-channel SCALE dequantization ported from the delivery
branches' `utils/py_utils/postprocess.py`, rdk_s @ 380e1a2). Activation
semantics are deliberately not part of this chain — whether a raw logit needs
a sigmoid or a dequantized output is already activated is a task-level fact
declared by each sample's `post_process`. The runner validates containers
only; the transform executes in `post_process` with the binding's quant
snapshot.

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
