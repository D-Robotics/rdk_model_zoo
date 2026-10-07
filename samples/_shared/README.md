# Shared sample utilities

These modules serve the samples, including ResNet, Ultralytics YOLO,
PaddleOCR and the two SAM pipelines. They do not define a global model runtime
or a global command.

`platforms.py` reads concrete identity from `docs/release/platforms.json`.
It checks boardinfo first, then X5's socinfo, then the observed device-tree
model. Exact `X5U` was observed on the two test X5 boards; `X5H` and `X5M`
are user-supplied aliases, not additional tested hardware. Explicit selection
is sufficient for preparation; local inference also needs matching identity.

`assets.py` reads the per-platform manifests at `docs/release/{x5,s}/models.yaml`.
The installed dependency is PyYAML (no Node/publisher or SDK dependency).
Existing manifests have sample IDs plus filenames. A reference in the form
`group:sample:filename` combines those fields. For example:

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

## Thin SDK session

`runtime.py:RuntimeSession(model_path, *, target)` is the narrow shared wrapper
around the board-side `hbm_runtime` SDK. Construction is SDK-free; `load`
applies `platforms.require_execution_target` before the SDK import and model
construction, so a requested/detected target mismatch fails with zero SDK
factory calls, and a failed load leaves no fake-success state (retry stays
possible). `run(inputs)` passes the SDK's native mappings through unchanged —
tensor names, input validation, output semantics and scheduling stay with each
sample's binding and runner. No `close`/context-manager behavior is assumed.
The integrated shared runner (`model_runner.py`) and the YOLO `ModelRunner`
use the session for the production loading boundary and then execute on the
loaded SDK object through their own validated calls (the same passthrough the
session's `run` performs) rather than routing execution through a second path.
`model_runner.RuntimeUnavailableError` is the session's exception type,
re-exported for established callers. Host tests patch the module's
`_default_runtime_factory` and the board identity source; they pin the
wrapper's call contract, not SDK behavior:
`python -m unittest discover -s samples/_shared/tests -p test_runtime_session.py`.

## Image bytes

`image.py:bgr_to_nv12_planes` is the single OpenCV I420 → interleaved NV12
implementation shared by the samples. `samples/_shared/tensor_io.py` wraps it
(and the classification samples reach it through that surface); the Ultralytics
YOLO, PaddleOCR, YOLOv5, YOLOE, FCOS, YOLO26 Depth, PP-LiteSeg, UNet and
UNetMobileNet runtimes import it directly. It accepts an even-sized uint8 BGR
image and returns contiguous uint8 `(1,H,W,1)` Y and `(1,H/2,W/2,2)` UV arrays.
The sample adapters preserve their existing validation and error behavior.
Resize/interpolation, packed-versus-split transport and tensor naming remain
sample-local because their contracts differ. NumPy/OpenCV imports are lazy.

The extraction preserves the operation order and chroma byte ordering of the
source implementations it was extracted from. Known-color and noncontiguous-input
tests cover the byte contract; sample tests and affected-board comparisons cover
consumers.

## Runtime metadata

`runtime_meta.py:RuntimeMetadata` reads only the public attributes exposed by
`hbm_runtime` objects and never imports a board SDK. A runtime that hosts
several models requires an explicit `model_name` selection, and any number of output tensors is
representable (single-output rules stay with each sample's binding).
`output_quants` carries the per-output quantization descriptors verbatim,
keyed by output name, so task contracts can inspect them without silently
dropping metadata. F32 outputs ignore vestigial quantization descriptors;
the raw values are already floats and must not be dequantized again. ResNet
and PaddleOCR re-export this class as their
`RuntimeMetadata`; each keeps its own contract checks on top.

`input_quants` also retains per-input descriptors for multi-input planning
models such as DiffusionDrive. Missing descriptors remain an empty mapping;
individual bindings decide whether they are required. Input/output descriptors
are projected without copying SDK objects. U16/U32 dtype spellings normalize to
uint16/uint32; this does not widen any existing sample's allowed dtype set.

`runtime_meta.py:metadata_evidence` projects a `RuntimeMetadata` (or a plain
mapping) into JSON-serialisable values without copying the SDK descriptors.
The projection keeps names, shapes, dtypes, strides and complete quant descriptors
(`quant_type`, `scale`, `zero_point`, `axis` plus further public attributes);
unknown objects raise instead of being stringified, and the metadata object is
never mutated. The sample evaluators use it for their `metadata` evidence.

## Declared output transforms

`quantization.py` implements the binding-declared chain from raw runtime
outputs to float32 values: `raw_f32` (float32 passthrough, including vestigial
descriptors; integer dtypes are rejected) and `dequant`
(per-tensor/per-channel SCALE dequantization). Activation
semantics are deliberately not part of this chain — whether a raw logit needs
a sigmoid or a dequantized output is already activated is a task-level fact
declared by each sample's `post_process`. The runner validates containers
only; the transform executes in `post_process` with the binding's quant
snapshot.

Affine SCALE decoding uses `(q - zero_point) * scale`. A scalar zero-point
broadcasts to every channel, a vector follows the declared channel axis,
and an empty zero-point means zero. For example, raw scores `[2, 4]`, scales
`[1, 3]`, and zero-point `7` decode to `[-5, -9]` (class 0). Symmetric
zero-points, vector offsets, per-tensor decoding, and non-SCALE passthrough
are also supported.

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
the SDK. Callers can inject `runtime` / `runtime_factory`. Metadata binding failure
discards the runtime; scheduling with no arguments remains a no-op. The runner
checks exact tensor names, physical input shape/dtype, output metadata and finite
values, then returns owned raw data. It never performs normalization, dequantization,
activation, argmax, geometry restoration, file IO or model downloads.

The result is one raw array. `physical_input` preserves the one-input API;
`physical_inputs` declares a name-to-shape/dtype mapping for split Y/UV input.
Exactly one contract must be supplied; missing/extra tensors fail before SDK run.
UNetMobileNet uses this split-input form. SAM/OCR, tracking state and
classification use their sample-specific contracts.

`single_array_runner.py:NamedArrayRunner` supplies the same transport for
multiple named raw outputs. LaneNet uses it to retain its float32 embedding,
int64 binary prediction and any observed auxiliary outputs without assuming their
order. The sample binding defines required roles; the transport validates every
observed name, shape, dtype and finite value, then returns owned arrays in a
mapping. `SingleArrayRunner` is a compatible one-output adapter over this base
and rejects additional outputs. The module path is stable for imports.

`dequantize_tensor(..., dtype="float64")` is an opt-in precision path for int32
score ordering; the default remains float32 for existing consumers. Explicit
NONE tensors are returned unchanged. This does not infer quantization metadata
or move decoding into the runtime runner.

## YOLOE-26 PF numerical kernels / 数值模块

[`yoloe26_geometry.py`](yoloe26_geometry.py) and
[`yoloe26_decode.py`](yoloe26_decode.py) preserve the fixed 4585-class,
640-square PF protocol from the S source. The [YOLOE Python task](../vision/yoloe/runtime/python/README.md),
[conversion guide](../vision/yoloe/conversion/README.md) and
[native C++ runtime](../vision/yoloe/runtime/cpp/README.md) use these modules.
The modules do not load models, download artifacts or interpret quantization descriptors.

The ten float32 NHWC outputs are class logits (4585), direct LTRB distances (4)
and mask coefficients (32) at strides 8, 16 and 32, followed by a
`[1,160,160,32]` prototype. Candidate selection uses deterministic Top-K with
strict confidence comparison, optional multiple labels and no NMS. Equal scores
retain source scale/anchor/class ordering. Integer or nonfinite arrays fail.

Geometry uses rounded resize dimensions, linear interpolation and padding 114.
Calibration RGB/255 and runtime BGR pixels share this preprocessing. The
immutable per-image context records actual horizontal and vertical scales. Mask
restoration interpolates logits before thresholding at zero, crops in model
space, removes padding, restores with nearest interpolation and returns owned
uint8 0/1 ROI masks. Inverse boxes use actual rounded scales; this intentionally
differs from the source's ideal common gain on images with rounding.

S public HBM artifacts declare quantized outputs. Prepare a floating-output
artifact through the conversion guide before using these float kernels.

这两个模块保留 S 源中的 4585 类、640 方形 PF 协议，由
[YOLOE Python 任务](../vision/yoloe/runtime/python/README_cn.md)、
[转换指南](../vision/yoloe/conversion/README_cn.md)与
[原生 C++ 运行时](../vision/yoloe/runtime/cpp/README_cn.md)使用。
模块不加载模型、不下载制品，也不执行反量化。输入是上述顺序的十个 NHWC float32 张量，
整数和非有限值显式拒绝。候选框采用直接 LTRB 解码和确定性 Top-K，支持单标签或多标签，
分数须严格大于阈值，不执行 NMS；平局沿用源中的尺度、anchor、类别顺序。

前处理按四舍五入尺寸缩放、线性插值并填充 114，校准 RGB/255 与运行 BGR
共享相同像素流程。每张图显式携带实际横纵缩放比例。掩码先插值 logits，
再按零阈值二值化、在模型坐标裁剪、去除 padding、最近邻还原，最后返回
独立拥有内存的 uint8 0/1 ROI。框坐标按实际缩放还原；存在尺寸取整时，
这是相对源代码理想统一缩放比例的有意修正。

已发布 S HBM 声明量化输出；使用这些浮点数值核前，应按转换指南准备浮点输出制品。

`text_metrics.py` scores saved Unicode transcripts without importing a model or
SDK. `edit_counts` returns Levenshtein substitution/deletion/insertion counts
with deterministic diagonal/deletion/insertion tie preference;
`score_transcripts` aggregates micro CER over unique utterance IDs. Characters
are Unicode code points, with whitespace, case and punctuation retained. Empty
reference corpora yield null CER while retaining insertion counts. These
metrics do not authenticate prediction provenance or execute inference. See
the [ASR evaluator](../speech/asr/evaluator/README.md) for a complete input schema
and runnable synthetic example.
