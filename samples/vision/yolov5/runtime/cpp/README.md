# YOLOv5 native C++ runtime

This directory is the native C++ counterpart of the unified YOLOv5 sample. It
keeps the X5 HB-DNN adapter and the S UCP adapter separate and shares only the
SDK-free pieces: tensor metadata gates (`yolov5_gate.*`), the numeric decoder
(`yolov5_decode.*`) and the evidence dump writer (`yolov5_dump.*`). The native
binary parses arguments, performs target-specific input/forward I/O, decodes the
three raw heads, writes an optional evidence dump, and renders through the
separate OpenCV visualizer. Publication facts are resolved by `launcher.py`
through `samples.vision.yolov5.runtime.python.model_binding`; the native binary
never guesses a layout from a file name.

<a id="supported-boards"></a>
## Supported boards

| Board | Status | Note |
| --- | --- | --- |
| X5 | supported-not-run | X5 HB-DNN source exists; this host has no X5 SDK, board or published model binary, so the native binary was not compiled or executed on hardware |
| S100 | supported-not-run | S UCP source exists; no board, SDK or `yolov5x_672x672_nv12.hbm` asset was available |
| S600 | supported-not-run | Same S source with the 64-byte BPU alignment macro; not built or run |
| S100P | not-supported | YOLOv5 has no published S100P asset; `--target s100p` is rejected |

Each adapter is compiled for exactly one target and the resulting binary refuses
a `--target` that differs from its compiled identity (see
[Interface and lifecycle](#interface-lifecycle)), because the S alignment macros
differ between S600 and the rest.

<a id="dependencies"></a>
## Dependencies

- CMake ≥ 3.16 and a C++17 compiler.
- OpenCV development headers and libraries (rendered output only).
- Horizon DNN headers under `/usr/hobot/include` and libraries under
  `/usr/hobot/lib`; the S target additionally links `hbucp`.
- The shared C++ helpers in `utils/c_utils` (`preprocess`, `postprocess`,
  `nn_math`), referenced by relative path in the CMake target.
- The launcher never installs packages, downloads a model, or reads the board
  identity: `--help`, `--list-models` and `--dry-run` run without an SDK.

<a id="build"></a>
## Build

The target is explicit; configuration never reads `/sys/class/boardinfo`.

```bash
# cwd: repository root
cmake -S samples/vision/yolov5/runtime/cpp -B samples/vision/yolov5/runtime/cpp/build/x5 -DYOLOV5_TARGET=x5
cmake --build samples/vision/yolov5/runtime/cpp/build/x5 --parallel

cmake -S samples/vision/yolov5/runtime/cpp -B samples/vision/yolov5/runtime/cpp/build/s100 -DYOLOV5_TARGET=s100
cmake --build samples/vision/yolov5/runtime/cpp/build/s100 --parallel
```

`YOLOV5_TARGET` must be `x5`, `s100`, `s100p` or `s600`; any other value is a
configure-time error. Exactly one adapter source is compiled per target, and
CMake defines both the SoC alignment macro (`SOC_S600` / `SOC_S100` /
`SOC_S100P`) and `YOLOV5_TARGET_NAME` used at runtime to reject a mismatching
`--target`. Expect a `yolov5_cpp` binary in the selected build directory.

<a id="run"></a>
## Run

Prerequisite: the model artifact prepared by
[`model/download.sh`](../../model/README.md) and the launcher's identity gate.

```bash
# cwd: repository root
# inspect the resolved publication fact without touching board or SDK
samples/vision/yolov5/runtime/cpp/run.sh --dry-run --target x5

# real run: exact asset id plus an external path, on a matching board
samples/vision/yolov5/runtime/cpp/run.sh --target x5 \
  --asset-id x5:yolov5:yolov5n_tag_v7.0_detect_640x640_bayese_nv12.bin \
  --model-path /absolute/yolov5n_tag_v7.0_detect_640x640_bayese_nv12.bin \
  --test-img /absolute/bus.jpg --dump-dir /tmp/yolov5-x5-dump

# S split-NV12 build
samples/vision/yolov5/runtime/cpp/run.sh --target s100 \
  --asset-id s:yolov5:s100/yolov5x_672x672_nv12.hbm \
  --dump-dir /tmp/yolov5-s100-dump
```

`--list-models --target <t>` prints the published assets for a target. An
external `--model-path` is accepted only together with the exact `--asset-id`
from the manifest. Expect `result.jpg` (or `--output <file>`) plus, when
`--dump-dir` is given, a `manifest.json` and one raw file per tensor.

<a id="parameters"></a>
## Parameters

| Parameter | Default | Description |
| --- | --- | --- |
| `--target` | `auto` | `x5`, `s100`, `s100p` or `s600`; `auto` resolves from board identity in the launcher |
| `--variant` | X5 `s-v2.0`, S `x-672` | Artifact variant; the X5 default is the fixed C++ source default |
| `--asset-id` | omitted | Required together with `--model-path`; must match the manifest exactly |
| `--model-path` | resolved manifest path | Model file; only valid with `--asset-id` |
| `--test-img` | sample test data | BGR input image |
| `--label-file` | none | One class label per line for rendering |
| `--output` | `result.jpg` | Rendered output image |
| `--dump-dir` | none | Directory for the machine-comparable evidence dump |
| `--score-thres` | `0.25` | Confidence threshold, finite value in `[0,1]` |
| `--nms-thres` | `0.45` | NMS IoU threshold, finite value in `[0,1]` |
| `--priority` | `0` | Scheduling priority; applied on S, rejected on X5 |
| `--bpu-core` | `-1` | BPU core (`-1` = runtime default); applied on S, rejected on X5 |

<a id="interface-lifecycle"></a>
## Interface and lifecycle

`yolov5::RuntimeOptions` is the native entry contract; `run_native` owns
target-specific model initialization, tensor allocation, cache operations,
synchronous forward and cleanup.

- X5: requires exactly one packed NV12 model with a compact
  `[1,3,640,640]` input and three native F32, `NONE`-quantized NHWC heads whose
  strides are exactly 8/16/32. The fixed X5 source writes the NV12 payload and
  reads the heads as flat compact buffers, so the gates require the reported
  aligned layout to equal the valid layout: a padded artifact is rejected with
  a precise reason instead of being misread, and the dump manifest records its
  `alignedShape`/`stride`/`alignedByteSize` for follow-up. The allocation must
  also cover a compact NV12 frame (`height*width*channels` floats for heads).
- S: requires one packed model with split `Y[1,672,672,1]` and
  `UV[1,336,336,2]` inputs and three metadata-described heads. The S SDK
  reports no `alignedShape`; the stored layout is `stride[]` plus
  `alignedByteSize`. Before any read, the dequantization gate proves native
  dtype, descriptor length and the addressing the fixed-source
  `dequantizeTensorS32` actually performs (element `(h,w,c)` at byte offset
  `(h*W + w)*stride[2] + c*stride[3]`): row padding (`stride[2]` larger than
  the compact row) and channel padding are genuinely supported and accepted,
  `stride[1]` must equal `width*stride[2]`, and the allocation must cover the
  stored (padded) extent. The raw dump keeps the full `alignedByteSize` extent
  with the strides in the manifest, so a padded run stays machine-comparable.
- Ownership: both adapters free only resources that were actually allocated, so
  a partially failed allocation never turns into a blind free. The X5 adapter
  releases the task and buffers through an RAII lease; the S adapter uses a
  guard that skips tensors whose `sysMem` was never assigned.
- The compiled build identity (`YOLOV5_TARGET_NAME`) must equal `--target`, so a
  binary built for one S alignment cannot run as another target.

Declared differences from the fixed sources (preserved, not silently unified):

- **Default X5 variant.** The fixed X5 C++ source defaults to the `s-v2.0`
  artifact; the unified Python runtime defaults to `n-v7.0`. The native
  launcher keeps the C++ source default when neither `--variant` nor `--asset-id`
  is given.
- **NMS.** X5 preserves the source `cv::dnn::NMSBoxes` behaviour per class: the
  score boundary is strictly greater than `--score-thres`, and each class is
  capped at `top_k = 300`. S preserves the source `nms_bboxes` behaviour: a
  score equal to `--score-thres` is kept and there is no per-class cap.
- **Preprocessing.** Both native adapters letterbox; the unified Python path
  uses stretch by default. This is a deliberate source-compatibility choice, not
  a claim that the two paths are numerically identical.
- **Scheduling.** The fixed S source forces `priority = 0`; the unified S
  adapter applies the caller's `--priority`/`--bpu-core` so the documented
  parameters are real. X5 has no verified HB-DNN mapping for these flags, so
  non-default values are rejected rather than silently ignored.
- **Non-finite scores.** The unified decoder drops non-finite confidence values;
  the source S decode keeps them. The unified behaviour is a declared fix.

<a id="results-interpretation"></a>
## Results interpretation

- Exit code `0` means the run completed; `2` means a rejected or failed run. On
  failure with `--dump-dir`, a manifest with `return_code` and `error` is still
  written so the failure is traceable.
- The rendered image is a convenience only. Machine comparison uses the dump:
  `manifest.json` binds `target`, `build_target`, `asset_id`, `model_path` and
  `image_path` with SHA-256 hashes, the observed input/output metadata
  (shape, dtype, quantization kind, scale length, `alignedByteSize`, the
  reported `stride[]`, and `alignedShape` where the SDK reports it — null
  otherwise), the effective parameters, the UTC timestamp, `argv`, `cwd` and
  `return_code`, and lists every raw and transformed tensor with its shape,
  byte count, file name and SHA-256.
- X5 raw and transformed tensors are the same native F32 heads; S raw tensors
  keep the full allocated extent (`alignedByteSize`, including row padding)
  and the transformed tensors are the dequantized floats, so a board
  comparison can check both stages and interpret padded layouts from the
  manifest strides.
- Board status (2026-09-24, coordinator evidence): the pre-remediation commit
  compiled and linked `rc=0` on a real X5 8GB and a first launcher inference
  returned `rc=0`; the same commit failed to compile on S100 because the S
  adapter used X5-only SDK spellings, which this round fixes per the on-board
  header evidence. No numerical board comparison, accuracy or performance
  claim is made from this tree; re-verification on the boards belongs to the
  coordinator. Host checks remain contract/decoder results only.
