English | [简体中文](README_cn.md)

# YOLOv5 native C++ runtime

This directory is the native C++ runtime of the YOLOv5 sample,
delivered as the ordinary five files: `inc/detect.hpp` + `src/detect.cpp` own
the detection model (SDK-free tensor gates, head decoder, S dequantizer, and
the X5 HB-DNN and S UCP backends behind compile-time target guards in the same
translation unit), `inc/cli.hpp` + `src/cli.cpp` own the command line, the
evidence dump writer and the rendered output, and `src/main.cpp` constructs
the model, runs `predict` and reports. Publication facts are resolved by
`launcher.py` through `samples.vision.yolov5.runtime.python.model_binding`; the
native binary never guesses a layout from a file name.

<a id="overview"></a>
## C++ inference

Run YOLOv5 detection with the X5 HB-DNN or S UCP adapter. The native application prepares NV12 input, decodes three detection heads and saves an annotated image.

<a id="directory"></a>
## Directory structure

```text
cpp/
├── inc/  # detect.hpp (model), cli.hpp (CLI)
├── src/  # main.cpp, cli.cpp, detect.cpp
├── CMakeLists.txt  # Explicit-target build
├── README.md  # English instructions
├── README_cn.md  # Chinese instructions
├── launcher.py  # Python script
└── run.sh  # Run the sample
```

<a id="supported-boards"></a>
## Supported boards

| Board | Artifact | SDK and input |
| --- | --- | --- |
| X5 | Published 640×640 `.bin` variant | X5 DNN SDK, compact NV12 |
| S100 | `x-672` `.hbm` | S100 UCP/DNN SDK, split NV12 |
| S600 | `x-672` `.hbm` | S600 UCP/DNN SDK, split NV12 |

All three boards above are supported through the listed artifacts and SDKs. After building, run the launcher on the board with its default case; accuracy and performance measurements are performed with the [evaluator guide](../../evaluator/README.md).

Each backend is compiled for exactly one target and the resulting binary refuses
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
configure-time error. One `src/detect.cpp` serves every target: the selected
backend is a compile-time guard, and CMake defines both the SoC alignment macro
(`SOC_S600` / `SOC_S100` / `SOC_S100P`) and `YOLOV5_TARGET_NAME` used at
runtime to reject a mismatching `--target`. Expect a `yolov5_cpp` binary in the
selected build directory.

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
| `--variant` | X5 `s-v2.0`, S `x-672` | Artifact variant; per-board default shown |
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

`yolov5::Yolov5` (declared in `inc/detect.hpp`) is the native model contract:
its constructor performs the build-identity and scheduling gates and loads the
runtime (RAII; a partially failed initialization frees exactly what it
allocated), and the public stages are `preprocess` → `infer` → `postprocess`
plus an explicit `predict` chain. `src/main.cpp` constructs the model visibly,
loads the image through the CLI (`inc/cli.hpp`) and passes caller-owned pixels:
`Input` is a BGR buffer plus its source geometry, `preprocess` converts it into
an owned `Prepared` NV12 payload without touching the SDK, `infer` validates
that payload at entry (exact model plane lengths and positive source geometry,
before any SDK allocation or copy) and uploads exactly its explicit argument —
never instance buffers a later call could rewrite — and `postprocess` only
decodes. `predict` chains the three stages and returns a `Prediction` (the
`Result` plus this call's `RunEvidence`), and `main` reports that returned
value through the CLI, which owns the dump and the rendered output. There are
no last-call accessors: every stage value is owned per call, so a returned
`Prediction` stays valid across later `predict` calls.

- X5: requires exactly one packed NV12 model with a compact
  `[1,3,640,640]` input and three native F32, `NONE`-quantized NHWC heads whose
  strides are exactly 8/16/32. The X5 backend writes the NV12 payload and
  reads the heads as flat compact buffers, so the gates require the reported
  aligned layout to equal the valid layout: a padded artifact is rejected with
  a precise reason instead of being misread, and the dump manifest records its
  `alignedShape`/`stride`/`alignedByteSize` for follow-up. The allocation must
  also cover a compact NV12 frame (`height*width*channels` floats for heads).
- S: requires one packed model with split `Y[1,672,672,1]` and
  `UV[1,336,336,2]` inputs and three metadata-described heads. The S SDK
  reports no `alignedShape`; the stored layout is `stride[]` plus
  `alignedByteSize`. Before any read, the dequantization gate proves native
  dtype, descriptor length and the addressing the dequantizer actually performs
  (element `(h,w,c)` at byte offset `(h*W + w)*stride[2] + c*stride[3]`):
  `stride[2]` must cover one full pixel (`channels` elements — the published
  S100 model's legal pixel padding `stride[2]=1024` for 255 channels is
  accepted, while a smaller value that makes pixels overlap is rejected),
  `stride[1]` must equal `width*stride[2]`, and the allocation must cover the
  exact last addressed byte with overflow-checked arithmetic. A scalar
  scale/zero-point descriptor (length 1) is accepted because the model's
  private dequantizer broadcasts it; the shared `c_utils`
  `dequantizeTensorS32` would index `scale_data[c]` out of bounds and is never
  given such a tensor. The raw dump keeps the full `alignedByteSize` extent
  with the strides and the full scale/zero-point arrays in the manifest, so a
  padded run stays machine-comparable.
- Ownership: both backends free only resources that were actually allocated, so
  a partially failed allocation never turns into a blind free. The X5 backend
  releases the task and buffers through an RAII lease; the S backend uses a
  guard that skips tensors whose `sysMem` was never assigned.
- The compiled build identity (`YOLOV5_TARGET_NAME`) must equal `--target`; build
  one binary per S alignment and run it on its matching target.

Python and C++ runtime behavior:

- **Default X5 variant.** The native launcher defaults to the `s-v2.0`
  artifact when neither `--variant` nor `--asset-id` is given; the Python
  runtime defaults to `n-v7.0`.
- **NMS.** X5 runs per-class `cv::dnn::NMSBoxes`: a score must be strictly
  greater than `--score-thres`, and each class keeps at most `top_k = 300`
  boxes. S runs `nms_bboxes`: a score equal to `--score-thres` is kept and
  there is no per-class cap.
- **Preprocessing.** Both native adapters letterbox; the Python runtime uses
  stretch by default.
- **Scheduling.** The S adapter applies the caller's `--priority` and
  `--bpu-core`. `--bpu-core` is a core *index* (`-1` = any, `0..3`) and
  is converted explicitly to the SDK's backend bitmask
  (`HB_UCP_BPU_CORE_0..3 = 1ULL<<0..3`, `HB_UCP_BPU_CORE_ANY = 1ULL<<7`);
  indices outside `-1..3` are rejected, and the raw index is never assigned to
  the backend field. X5 has no verified HB-DNN mapping for these flags, so
  non-default values are rejected rather than silently ignored.
- **Non-finite scores.** The decoder drops non-finite confidence values.

<a id="results-interpretation"></a>
## Results interpretation

- Exit code `0` means the run completed; `2` means a rejected or failed run. On
  failure with `--dump-dir`, a manifest with `return_code` and `error` is still
  written so the failure is traceable.
- The rendered image is a convenience only. Machine comparison uses the dump:
  `manifest.json` binds `target`, `build_target`, `asset_id`, `model_path`,
  `image_path` and the running `binary_path` with SHA-256 hashes, the observed
  input/output metadata (shape, dtype, quantization kind, scale length, the
  full scale/zero-point values, `quantizeAxis`, `alignedByteSize`, the
  reported `stride[]`, and `alignedShape` where the SDK reports it — null
  otherwise), the effective parameters, the UTC timestamp, `argv`, `cwd` and
  `return_code`. Tensor payloads are written one file per stage under
  `input/`, `raw/` and `transformed/` subdirectories with shape, byte count,
  file name and SHA-256, so the raw and transformed bytes of one output can
  never overwrite each other.
- The input files hold the buffers actually submitted with the inference (the
  compact NV12 payload on X5, the compact Y/UV plane payload on S — exactly
  the valid row bytes the upload writes at the stride pitch);
  uninitialized padding bytes are deliberately not dumped. X5 raw and
  transformed tensors are the same native F32 heads; S raw tensors keep the
  full allocated extent (`alignedByteSize`, including pixel padding) and the
  transformed tensors are the dequantized floats, so a board comparison can
  check both stages and interpret padded layouts from the manifest strides.
  A dump records what this binary produced; the board evaluator performs any
  cross-runtime assessment separately.
