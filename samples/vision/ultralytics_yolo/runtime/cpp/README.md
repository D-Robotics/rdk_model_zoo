English | [简体中文](README_cn.md)

# Ultralytics YOLO C++ runtime

[Python](../python/README.md)

<a id="overview"></a>
## C++ inference

Run the supported Ultralytics detection, segmentation, pose, classification or
oriented-box model on its matching board. One executable covers all five tasks
(`--task detect|segment|pose|classify|obb`); the native backend stack and the
packed/split NV12 input protocol are selected from model metadata.

<a id="directory"></a>
## Directory structure

```text
cpp/
├── CMakeLists.txt  # Board build (C++17, YOLO_TARGET=auto|x5|s100|s100p|s600)
├── run.sh          # Out-of-tree build + launch, forwards all arguments
├── inc/
│   ├── yolo.hpp            # SDK-free vocabulary: decode math, plans, task types
│   ├── backend.hpp         # Shared DNN backend: stack probe, infer, RAII owners
│   ├── cli.hpp             # Inline option parsing, benchmark math and entries
│   └── imagenet_labels.hpp # 1000-entry ImageNet label order (classification)
├── src/
│   ├── main.cpp     # Constructs the task model, calls predict, reports
│   ├── cli.cpp      # Image loading, rendering, benchmark drivers
│   ├── backend.cpp  # SDK lifecycle: infer_sync, input probe, NV12 upload
│   ├── detect.cpp   # YoloDetect model (LTRB + DFL heads)
│   ├── segment.cpp  # YoloSegment model (masks, prototypes)
│   ├── pose.cpp     # YoloPose model (17 COCO keypoints)
│   ├── classify.cpp # YoloClassify model (1000-logit Top-K)
│   └── obb.cpp      # YoloObb model (direct LTRB + angle, rotated NMS)
└── test/  # Host unit tests (ctest; fake X5/UCP SDK doubles in test/fake_dnn_io)
```

<a id="supported-boards"></a>
## Supported protocols

The runtime selects protocols from model metadata: packed NV12 `.bin` input for
X5 and split Y/UV `.hbm` input for S100/S100P/S600. Build once per board
target; the same source compiles against both stacks.

| Task | Implemented protocol | Limits |
|---|---|---|
| detect | YOLO26 four-channel LTRB; YOLO11-family 64-channel DFL | Explicit head selection |
| pose | DFL/LTRB boxes and corresponding keypoint encoding | Functional reference |
| segment | DFL/LTRB boxes and mask coefficients/prototypes | Functional reference |
| classify | One unquantized FLOAT32 vector of 1000 logits | Stride-aware Top-5, no result image |
| obb | YOLO26 direct-LTRB boxes plus a per-cell angle, nine outputs | Square stride-32 input, DOTA classes (default 15) |

Custom class/model layouts must match the labels and dimensions expected by
the selected task. Use the linked Python entry for the YOLOv10 NMS-free task
interface.

<a id="dependencies"></a>
## Dependencies

Board builds require CMake ≥3.16, a C++17 compiler, OpenCV development files
and the matching board DNN SDK. `YOLO_TARGET=auto` (default) first checks
`/usr/include/dnn/hb_dnn.h` and `/usr/lib`; otherwise it uses
`/usr/include/hobot`, `/usr/include/hobot/dnn`, `/usr/hobot/include`,
`/usr/hobot/lib` and links `hbucp`. Pin `-DYOLO_TARGET=x5|s100|s100p|s600` to
select explicitly; the build never reads the host SoC. These files come from
the matching board image/SDK; use the SDK version supplied for the target
image. Host tests additionally use CTest ≥3.20 (`ctest --test-dir`).

<a id="build"></a>
## Build

Run from the sample root (`samples/vision/ultralytics_yolo`) on the target
board. `run.sh` configures an out-of-tree build under `runtime/cpp/build/`,
builds it and executes the binary; every argument is forwarded:

```bash
bash runtime/cpp/run.sh --task detect model/yolo26n_detect_bayese_640x640_nv12.bin \
  test_data/bus.jpg /tmp/cpp-result.jpg
```

Equivalent manual build:

```bash
cmake -S runtime/cpp -B /tmp/ultralytics-cpp -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/ultralytics-cpp -j2
/tmp/ultralytics-cpp/ultralytics_yolo_cpp --task detect ...
```

<a id="run"></a>
## Run

Each task has its own model, image and result-path defaults, so a bare
`--task` invocation runs the reference flow from the sample root. This X5
path explicitly prepares a model and produces a detection image:

```bash
bash model/download_model.sh --platform x5 --family yolo26 --task detect --model-size n
bash runtime/cpp/run.sh --task detect \
  model/yolo26n_detect_bayese_640x640_nv12.bin \
  test_data/bus.jpg /tmp/cpp-result.jpg
```

For the other tasks, first prepare the corresponding artifact using
[model instructions](../../model/README.md), then pass actual paths. S
requires matching-march `.hbm` files; renaming X5 files does not convert them.

```bash
runtime/cpp/run.sh --task classify /models/classification.bin test_data/zebra_cls.jpg
runtime/cpp/run.sh --task pose /models/pose.bin test_data/bus.jpg /tmp/cpp-pose.jpg
runtime/cpp/run.sh --task segment /models/segmentation.bin test_data/bus.jpg /tmp/cpp-segment.jpg
runtime/cpp/run.sh --task obb /models/obb.bin test_data/dota.jpg /tmp/cpp-obb.jpg
```

<a id="parameters"></a>
## Parameters

All tasks accept positional `model`, `image` and (except classify) `result`
paths plus the kebab-case options below, matching the Python CLI spelling.
Detect/pose/segment draw a result image; classify prints Top-K only.

| Option | Default | Meaning |
|---|---|---|
| `--task` | `detect` | Task runtime: detect, segment, pose, classify or obb |
| `--head` | `auto` | auto/dfl/ltrb; detect only |
| `--score-thres` | `0.25` | Score cutoff (raw-logit comparison internally) |
| `--nms-thres` | `0.7` detect, `0.45` segment/pose, `0.2` obb | NMS IoU threshold |
| `--kpt-conf-thres` | `0.5` | Pose keypoint confidence |
| `--topk` | `5` | Classification Top-K |
| `--classes` | `15` | OBB class channels |
| `--angle-sign` | `1` | OBB angle sign convention |
| `--angle-offset` | `0` | OBB angle offset in degrees |
| `--no-regularize` | `false` | Keep OBB boxes unregularized |
| `--resize-type` | `1` | 0 stretch, 1 letterbox (all tasks) |
| `--opencv-threads` | `all` | 0/all: online CPU count; explicit positive count supported |
| `--benchmark` | `false` | Bounded end-to-end measurement (all tasks) |
| `--warmup` | `20` | Warmup frames per round |
| `--runs` | `200` | Timed frames per round |
| `--rounds` | `3` | Number of rounds |
| `--pipeline-streams` | `1` | 1 or 2 independent complete concurrent pipelines |
| `--json` | `empty` | Write aggregate benchmark JSON to this path |
| `--runtime-source-sha256` | `empty` | Provenance hex recorded in the benchmark JSON |
| `--executable-sha256` | `empty` | Provenance hex recorded in the benchmark JSON |
| `--no-save` | `false` | Skip drawing and saving the validation image |
| `--help / -h` | `false` | Print CLI help |

Per-task defaults when positionals are omitted: detect uses
`model/yolo26n_detect_bayese_640x640_nv12.bin`, `test_data/bus.jpg` and
`cpp_result.jpg`; segment and pose use the corresponding
`source/reference_bin_models/{seg,pose}/yolo11n_*_bayese_640x640_nv12.bin`
with `segment_result.jpg`/`pose_result.jpg`; classify uses
`source/reference_bin_models/cls/yolo11n_cls_bayese_224x224_nv12.bin` and
`test_data/zebra_cls.jpg`; obb uses `yolo26n_obb_640x640_nv12.bin`,
`test_data/dota.jpg` and `obb_result.jpg`. Paths are relative to the caller.

<a id="interface-lifecycle"></a>
## Interfaces and resource lifetime

This is a reference executable plus a readable model API, not a stable library
contract identical to Python. `main.cpp` constructs the named model
(`yolo::YoloDetect`, `YoloSegment`, `YoloPose`, `YoloClassify`, `YoloObb`)
from a `Config`,
fills a caller-owned `Input` (raw BGR pixels plus source geometry), calls
`predict` and hands the returned `Prediction` to the CLI reporter. Each model
owns its lifecycle: constructor loads the runtime (model handle, input
protocol probe, output binding), `preprocess → infer → postprocess` are the
model's own stages, and every call owns its input/output buffers — no hidden
last-call state. The CLI owns options, image loading, rendering and the
benchmark.

Each concurrent benchmark stream owns its runtime context and tensors; do not
reuse buffers across unfinished requests. The shared DNN backend
(`inc/backend.hpp` + `src/backend.cpp`) owns the SDK surface: the X5/UCP stack
probe, `infer_sync`, the NV12 input owner, `PackedModelOwner`/
`OutputTensorOwner` and the stride-validated output reader. It also exposes
`infer_tensors_sync` for out-of-sample consumers (ASR/Paraformer) with more
than two already validated input tensors; the image-facing `infer_sync`
retains its one/two-input restriction. Integrations must check SDK return
codes, strides and valid shapes, not merely copy the inference call.

<a id="classification-contract"></a>
## Classification output contract

The classification runtime accepts exactly one model and one output containing
1000 finite FLOAT32 logits, with quantization `NONE`. Supported shapes are a
rank-one `(1000,)` vector or rank 2–4 with batch one, one 1000-class axis and
all other axes singleton, for example `(1,1000)`, `(1,1000,1,1)` or
`(1,1,1,1000)`. It does not assume that axis 1 always holds classes. Other
class counts, integer outputs, flattened spatial maps and batched outputs fail
explicitly.

The output requires a positive physical allocation size. X5 reads class
strides from aligned dimensions; S reads byte strides. Padding is skipped, and
the last class must fit in the allocation before any read. Stable softmax uses
a double accumulator; Top-K returns probability descending, then class ID
ascending for exact equal probabilities. Labels follow the 1000-entry
ImageNet order in `inc/imagenet_labels.hpp`; classification math lives in
`inc/yolo.hpp`. No manual dequantization is performed.

Model and output owners release acquired resources on early returns and C++
exceptions. Allocation and output-cache invalidation failures stop decoding.
Malformed descriptors/nonfinite logits produce a nonzero exit with a
diagnostic.

Classification preprocessing defaults to **letterbox with gray 127 padding**;
Python classification on S and Python YOLO26 use stretch. Pass
`--resize-type 0` to switch the C++ runtime to stretch without rebuilding.

<a id="pose-segment-output-contract"></a>
## Pose and segmentation output contract

Both tasks require square input dimensions divisible by 32, batch one and
unquantized FLOAT32 NHWC outputs. Roles are matched by spatial shape and
channels, not output enumeration: strides 8/16/32 each contain classes
(pose 1, segment 80), boxes (4 direct LTRB or 64 DFL) and extras (pose 51
keypoint values, segment 32 mask coefficients). All three scales must use the
same box encoding. Segment adds one stride-4, 32-channel NHWC prototype.
Missing/duplicate roles, mixed encodings, integer/SCALE tensors, unsupported
layouts and nonfinite values fail explicitly. In particular this C++ prototype
path does **not** accept Python's NCHW prototype alternative; use the matching
NHWC export or the Python entry.

X5 aligned dimensions and S byte strides determine physical reads. After cache
invalidation, valid values are copied to owned compact NHWC vectors, skipping
padding. This adds a temporary host copy of the valid outputs. Allocation and
cache errors abort; acquired output/model resources release on all exit paths.
DFL and direct-distance math live in `inc/yolo.hpp` (`decode_box_dfl`,
`decode_box_ltrb`) rather than private copies. The existing keypoint
equations, NMS and rendering policies remain unchanged.

Both tasks discard boxes crossing the model-input boundary. Segment uses
class-agnostic NMS and renders a **model-input-sized** three-panel image
(detections, colored mask, combined), with total width `3 * input_width`. Pose
renders on the original image using its resize/padding arithmetic. A failed
image save produces a nonzero exit.

<a id="obb-output-contract"></a>
## Oriented-box output contract

The oriented-box runtime (`--task obb`) requires a square input divisible by
32, batch one and exactly nine unquantized FLOAT32 NHWC outputs: per stride
8/16/32 one class map (`--classes`, default 15 DOTA classes), one 4-channel
direct-LTRB box map and one 1-channel angle map. Roles are matched by spatial
shape and channel count; class counts of 1 or 4 collide with the angle/box
roles and are rejected, so pass an explicit `--classes` for such layouts.
Decoding follows `runtime/python/obb.py`: LTRB distances scaled by the
stride around the cell centre, angle scaled by `--angle-sign` and shifted by
`--angle-offset` (degrees). Boxes are regularized by default (width ≥ height,
angle wrapped); `--no-regularize` keeps the raw geometry.

The platform policy also matches the Python entry: X5 wraps angles to
[-π/2, π/2), runs per-class greedy rotated NMS (`--nms-thres`, default 0.2)
and clips restored boxes to the source image; the S series runs class-agnostic
rotated NMS and keeps unclipped geometry. Letterbox restore uses the realized
integer-rounded resize ratio, not the ideal scale. Nonfinite decoded values
fail explicitly; `--angle-sign`/`--angle-offset` must be finite.

<a id="results-interpretation"></a>
## Results and verification

Boxes, masks and keypoints are drawn into result images; classification prints
classes and probabilities. Check the process return code and inspect the
output image. Classification IDs index the model's built-in class order.
Dataset scoring is described in the evaluator guide.

Bounded benchmark, available for every task:

```bash
runtime/cpp/run.sh --task detect \
  model/yolo26n_detect_bayese_640x640_nv12.bin test_data/bus.jpg /tmp/cpp-result.jpg \
  --benchmark --warmup 20 --runs 200 --rounds 3 \
  --pipeline-streams 1 --opencv-threads all \
  --score-thres 0.25 --nms-thres 0.7 --no-save --json /tmp/yolo-e2e-cpp.json
```
Timing starts with an in-memory BGR image and ends with restored results:
resize/letterbox, NV12 conversion, copy/cache operations, BPU and decode/NMS
are included; model load, file I/O, drawing and saving are excluded. Set
`--pipeline-streams 2` for two complete pipelines. Throughput is total
completed frames over shared wall time; latency is per request. OpenCV thread
count is independent of stream count, with no CPU-affinity restriction. Keep
C++/Python, runtime-only/end-to-end and single/multistream measurements
separate.

The process performs exactly one validation predict and holds exactly one
runtime context per stream (stream 0 reuses the model constructed for
validation). The JSON record reports the task's `output_kind`
(`detections`, `instance_masks`, `pose_instances`, `topk_predictions`,
`rotated_boxes`) with `outputs_per_frame`, `resize_type` when applicable, the
angle options for rotated boxes, and score/NMS for detection-style tasks.
`--runtime-source-sha256`/`--executable-sha256` (64 hex digits) record optional
provenance in the same record.

The shared input owner validates metadata before allocating. Packed X5 NV12
accepts RGB-shaped NCHW/NHWC descriptors only when the physical shape is
compact; padded packed storage is explicitly rejected. Split Y/UV accepts
exact batch-one geometry and unquantized byte planes. Dynamic `-1`
strides/capacity are resolved from the actual row pitch; zero/overlapping
strides, insufficient allocation and overflow are rejected. `upload_planes`
takes exact-length compact Y and interleaved UV buffers, writes the allocated
row pitch and cleans both caches. Existing I420 `upload` delegates to that
path. Uploads before successful allocation or with a different plan fail;
partial allocation releases all acquired buffers.

Synchronous inference requires a non-null task handle, releases returned tasks
on creation/submission/wait failures and preserves the first error code. UCP
submission selects `HB_UCP_BPU_CORE_ANY`. Build on the target with matching
DNN/UCP SDK headers and libraries. If model binding rejects an artifact, check
its target, task, layout, dtype and output head against the selected program.

The shared `OutputTensorOwner` retains an acquired buffer for cleanup when an
SDK allocation returns an error with a non-null address, and rejects success
with a null address.
