# YOLO detection contract

This document records the contracts implemented by the unified Python DFL and
YOLO26 direct-LTRB detection paths. It covers the observed YOLOv8n artifacts
used for the P1 pilot and the observed YOLO26n artifacts used for P2 Task 1.
It is not a claim that every model asset in the Model Zoo has been mapped or
board validated.

## Runtime boundary

`YoloDetect` owns image geometry, DFL decoding, suppression, and the public
result. `ModelRunner` owns one loaded runtime model and one `ModelBinding`.
The task can be constructed with `YoloDetect(config, runner=...)` for a
callable runner supplied by a test or another supported runtime. The legacy
`YoloDetect(config)` constructor uses `build_runner(config)`.

The binding is created once, from the selected artifact and the metadata read
from that artifact. A complete binding requires:

* one packed NV12 input on X5, or named `y` and `uv` inputs on S100/S100P/S600;
* batch one, positive even input dimensions, and `uint8` transport data;
* one reviewed output protocol: three classification tensors plus three DFL
  tensors, or three classification tensors plus three direct four-channel LTRB
  tensors;
* NHWC output shapes that match the declared grid and channel count;
* floating point output data. The reviewed DFL contract may use integer output
  only when the artifact contract supplies an explicit scalar `scale` and
  optional `zero_point`. The reviewed LTRB contract has no quantization
  protocol and rejects integer or quantized output metadata.

An unknown shape, dtype, output role, or quantization parameter stops binding
before inference. A filename, output enumeration position, or `.bin`/`.hbm`
suffix is not a protocol declaration.

## Logical DFL protocol

The P1 contract is:

| Role | Semantic shape | Meaning |
| --- | --- | --- |
| `cls_<stride>` | `[1, Hs, Ws, C]` | class logits, before sigmoid |
| `box_<stride>` | `[1, Hs, Ws, 4, R]` | DFL logits, represented physically as `[1, Hs, Ws, 4R]` |

The pilot uses `C=80`, `R=16`, and strides `8, 16, 32`. For a model input
`(H, W)`, each feature grid is `(H/stride, W/stride)`; the height and width
are checked independently. Classification scores use one sigmoid and the
largest class score per cell. DFL uses one softmax per side and the expected
bin index. The resulting boxes are in model-input pixel coordinates until
`ImageTransform` maps them to the original image.

The task's public return value is the tuple-compatible
`DetectionResult(boxes_xyxy, scores, class_ids)`. `boxes_xyxy` has shape
`(N, 4)` and original-image pixel coordinates; scores and IDs have shape
`(N,)`. Empty detections are returned as typed empty arrays.

The observed detector variants use class-wise NMS. The existing task defaults
remain X5 `0.70` and S `0.45`; those values come from the selected platform
profile and are not inferred from the tensor names.

## Observed pilot artifact mapping

The following mapping is grounded in the runtime metadata captured in
`docs/releases/unified-migration/evidence/*-metadata.log`. The compiler names
are opaque, so the binding discovers the same map from shape and dtype and
validates it on every model load.

| Target | Input roles | `cls_8` / `box_8` | `cls_16` / `box_16` | `cls_32` / `box_32` |
| --- | --- | --- | --- | --- |
| X5 | `image=images` (`[1,3,640,640]`, NV12) | `output0` / `326` | `334` / `342` | `350` / `358` |
| S100 | `y=images_y`, `uv=images_uv` (NHWC U8) | `output0` / `330` | `338` / `346` | `354` / `362` |
| S100P | `y=images_y`, `uv=images_uv` (NHWC U8) | `output0` / `330` | `338` / `346` | `354` / `362` |
| S600 | `y=images_y`, `uv=images_uv` (NHWC U8) | `output0` / `330` | `338` / `346` | `354` / `362` |

For all four records, the output shapes are respectively `80x80`, `40x40`,
and `20x20` with channels `80`, `64`, `80`, `64`, `80`, `64`; the SDK reports
`hbDNNDataType.F32`. The S input records are `[1,640,640,1]` and
`[1,320,320,2]`, both `hbDNNDataType.U8`.

The binding expresses this finite, reviewed DFL protocol in
`model_binding.py`; shape matching only identifies the six opaque physical
names after the protocol is selected and does not prove that an arbitrary
model has YOLO semantics. The current pilot records and board evidence are
YOLOv8n only. The unified entrypoint still exposes the existing DFL-compatible
families such as its default YOLO11 selection; each selected artifact must
provide complete metadata and pass this binding at load time. Other output
protocols, custom local files, and NMS-free or already decoded outputs need
their own reviewed contract.

## Logical YOLO26 direct-LTRB protocol

The P2 Task 1 contract is a sibling of the DFL contract, not a DFL variant:

| Role | Semantic shape | Meaning |
| --- | --- | --- |
| `cls_<stride>` | `[1, Hs, Ws, C]` | class logits, before sigmoid |
| `box_<stride>` | `[1, Hs, Ws, 4]` | direct left, top, right, bottom distances in cell units |

The reviewed YOLO26n artifacts use `C=80`, strides `8, 16, 32`, NHWC
classification and LTRB tensors, and `F32` output metadata. The decoder takes
the maximum class in raw-logit space, applies sigmoid only to the selected
class, converts the four distances using the cell-center grid, then applies
the shared class-wise NMS helper. It does not apply DFL softmax or infer a
four-channel tensor as DFL bins. Unsupported NCHW, flat, integer, quantized,
or alternate output protocols fail binding or decoding.

The YOLO26n compiler output names are opaque and differ by target. The
observed mapping is:

| Target | Input roles | `cls_8` / `box_8` | `cls_16` / `box_16` | `cls_32` / `box_32` |
| --- | --- | --- | --- | --- |
| X5 | `image=images` (`[1,3,640,640]`, NV12) | `output0` / `580` | `594` / `602` | `616` / `624` |
| S100, S100P, S600 | `y=images_y`, `uv=images_uv` (`[1,640,640,1]`, `[1,320,320,2]`, U8) | `output0` / `609` | `623` / `631` | `645` / `653` |

This finite map is used as a shape-and-dtype checked protocol binding; shape
matching identifies the opaque tensor roles after the LTRB contract is chosen
but does not establish arbitrary model semantics. The current implementation
and board comparison cover the YOLO26n records above. Other YOLO26 tasks and
assets remain on their existing paths until separately audited.

## Geometry

`resize_with_transform` records the source size, final model size, actual
resized size after integer truncation, each padding side, and the actual X/Y
scales. `inverse_boxes` uses those recorded values. This matters for a
letterbox such as a `3x7` image into `10x10`: the resized content is `4x10`
after truncation, with three pixels of padding above and below, so the inverse
must use `4/3` vertically rather than the ideal floating point scale.

## Source-symbol mapping

The old canonical detector in
`runtime/python/yolo_detect.py` loaded the SDK and selected
`output_names[2*i]`/`output_names[2*i+1]` inside `YoloDetect.__init__`, converted
images in `pre_process`, called `model.run` in `forward`, decoded and applied
NMS in `post_process`, then restored coordinates in `predict`.

The maintained symbols now map as follows:

| Previous symbol or responsibility | Maintained symbol |
| --- | --- |
| `YoloDetect.__init__` model load | `model_runner.build_runner` → `ModelRunner.from_selection` |
| runtime input metadata and NV12 protocol | `model_binding.bind_model` → `tensor_io.bind_nv12_inputs` |
| positional output pairing | `model_binding._bind_output_roles` → `OutputBinding` role map |
| `YoloDetect.pre_process` | `geometry.resize_with_transform` + `InputBinding.build` |
| `YoloDetect.forward` | injected `ModelRunner` callable |
| `filter_classification` / `decode_boxes` path | `decode.decode_dfl` |
| `YOLO26Runtime` LTRB postprocess | `decode.decode_ltrb` through `YOLO26Detect` |
| `scale_coords_back` path | `geometry.inverse_boxes` |
| tuple result | `DetectionResult`, a `NamedTuple` retaining tuple unpacking |

The platform compatibility wrappers
`platforms/x5/.../ultralytics_yolo_det.py` and
`platforms/s/.../yolo_detect.py` continue to provide their historical class
and configuration names and forward to this task. They do not define another
detector algorithm.

`YOLO26Detect` uses the same shared image preparation, runner call, transform
recording, and result orchestration helpers as `YoloDetect`; only the
protocol-specific decoder and binding contract differ.

## Board validation scope

Fixed-input old/new YOLOv8n comparisons passed on X5 8GB/4GB, S100, S100P,
and S600, using the same per-target artifact and bus image as the original
source. The native command, compatibility wrapper, explicit scheduling,
returned-result lifetime across another inference, and wrong-target rejection
were also exercised. See the [validation record](../../../docs/releases/unified-migration/2026-09-16-pilot-validation.md)
for exact source snapshots, artifacts, runtime versions, tolerances, and logs.
This does not certify every DFL family/scale, performance, dataset accuracy,
SDK raw-buffer ownership beyond the returned task results, or other YOLO tasks.

P2 Task 1 also compared YOLO26n direct-LTRB on the same five targets. Boxes,
scores, class IDs and ordering matched each original-source baseline exactly;
native entry and returned-result lifetime passed. The same candidate reran DFL
and available ResNet18 board regressions. See the [P2 validation record](../../../docs/releases/unified-migration/2026-09-16-p2-validation.md)
for the distinct source snapshot, 95 host tests and independent review. Other
YOLO26 tasks and model scales remain outside this result.
