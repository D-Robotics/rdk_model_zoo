# Ultralytics YOLO C++ runtime

[简体中文](README_cn.md) · [Python](../python/README.md)

<a id="overview"></a>
## C++ inference

Use this directory for c++ inference.

<a id="directory"></a>
## Directory structure

```text
cpp/
├── classify/  # Files for classify
├── common/  # Files for common
├── detect/  # Files for detect
├── pose/  # Files for pose
├── segment/  # Files for segment
├── test/  # Files for test
├── README.md  # English instructions
└── README_cn.md  # Chinese instructions
```

<a id="supported-boards"></a>
## Supported protocols

The C++ entries select protocols from model metadata: packed NV12 `.bin` input for X5 and split Y/UV `.hbm` input for S100/S100P/S600. Build and run each task on its target board using the sections below.

| Program | Implemented protocol | Limits |
|---|---|---|
| detect | YOLO26 four-channel LTRB; YOLO11-family 64-channel DFL | Explicit head selection and benchmark |
| pose | DFL/LTRB boxes and corresponding keypoint encoding | Functional reference, no benchmark CLI |
| segment | DFL/LTRB boxes and mask coefficients/prototypes | Functional reference, no benchmark CLI |
| classify | One unquantized FLOAT32 vector of 1000 logits | Stride-aware Top-5, no result image |

Custom class/model layouts must match the labels and dimensions expected by the selected C++ task. Use the linked Python entry for the YOLOv10 NMS-free task interface.

<a id="dependencies"></a>
## Dependencies

Board builds require CMake/CTest ≥3.20 (including `ctest --test-dir`), a C++11 compiler, OpenCV development files and the matching board DNN SDK. CMake first checks `/usr/include/dnn/hb_dnn.h` and `/usr/lib`; otherwise it uses `/usr/include/hobot`, `/usr/include/hobot/dnn`, `/usr/hobot/include`, `/usr/hobot/lib` and links `hbucp`. These files come from the matching board image/SDK; use the SDK version supplied for the target image.

<a id="build"></a>
## Build

Run from the repository root on the target board. Each task owns its CMakeLists.txt; this directory has no top-level CMake project or run.sh.

```bash
for task in detect classify pose segment; do
  cmake -S "samples/vision/ultralytics_yolo/runtime/cpp/$task" \
    -B "/tmp/ultralytics-cpp-$task" -DCMAKE_BUILD_TYPE=Release
  cmake --build "/tmp/ultralytics-cpp-$task" -j2
done
```

<a id="run"></a>
## Run

This X5 path explicitly prepares a model and produces a detection image, still from the repository root:

```bash
bash samples/vision/ultralytics_yolo/model/download_model.sh \
  --platform x5 --family yolo26 --task detect --model-size n
/tmp/ultralytics-cpp-detect/ultralytics_yolo_detect \
  samples/vision/ultralytics_yolo/model/yolo26n_detect_bayese_640x640_nv12.bin \
  samples/vision/ultralytics_yolo/test_data/bus.jpg /tmp/cpp-result.jpg
```
For the other tasks, first prepare the corresponding artifact using [model instructions](../../model/README.md), then replace `/models/*.bin` with actual paths. S requires matching-march `.hbm` files; renaming X5 files does not convert them.

```bash
/tmp/ultralytics-cpp-classify/ultralytics_yolo_classify \
  /models/classification.bin samples/vision/ultralytics_yolo/test_data/zebra_cls.jpg
/tmp/ultralytics-cpp-pose/ultralytics_yolo_pose \
  /models/pose.bin samples/vision/ultralytics_yolo/test_data/bus.jpg /tmp/cpp-pose.jpg
/tmp/ultralytics-cpp-segment/ultralytics_yolo_segment \
  /models/segmentation.bin samples/vision/ultralytics_yolo/test_data/bus.jpg /tmp/cpp-segment.jpg
```

<a id="parameters"></a>
## Parameters

Detect/pose/segment take positional model, image and result paths; classify takes model and image only. Pose/segment/classify do not implement Python-style `--platform`, `--model-path` or general `--help`; these strings would be treated as positional paths. Pass model, image and result paths explicitly.

Only **detect** accepts these options:

| Option | Default | Meaning |
|---|---|---|
| `--head` | `auto` | auto/dfl/ltrb; detect only |
| `--score` | `0.25` | Detection score cutoff |
| `--nms` | `0.7` | Detection NMS IoU; does not adopt the Python S default |
| `--resize-type` | `1` | 0 stretch, 1 letterbox |
| `--benchmark` | `false` | Bounded end-to-end measurement |
| `--warmup` | `20` | Warmup frames per round |
| `--runs` | `200` | Timed frames per round |
| `--rounds` | `3` | Number of rounds |
| `--pipeline-streams` | `1` | Independent complete concurrent pipelines |
| `--opencv-threads` | `0` | 0/all: online CPU count; explicit positive count supported |
| `--json` | `empty` | Write aggregate benchmark JSON to this path |
| `--no-save` | `false` | Skip drawing and saving validation image |
| `--help / -h` | `false` | Print detection CLI help |

Pose/segment source constants are score=0.25 and NMS=0.45; pose point threshold is 0.5 and classify Top-K is 5. These are not CLI flags. Detect defaults to `yolo26n_detect_bayese_640x640_nv12.bin`, `bus.jpg` and `cpp_result.jpg`, relative to the caller.

<a id="interface-lifecycle"></a>
## Interfaces and resource lifetime

These are standalone reference executables, not a stable library API identical to Python. `DetectRuntime` in `detect/main.cc` owns model/tensors; `common/dnn_io` binds input, copies NV12 and cleans the input cache. After execution, output cache handling precedes decode/NMS. Detect and pose
restore coordinates to the original image; segment renders in model-input space,
as detailed below. Tensors/models must not be released before requests complete.

Each concurrent stream owns its runtime context and tensors; do not reuse buffers across unfinished requests. Geometry, head probing/decode and benchmark bookkeeping live in `common/`; pose/segment/classify retain their own main programs and resource flow. Integrations must check SDK return codes, strides and valid shapes, not merely copy the inference call.

<a id="classification-contract"></a>
## Classification output contract

The classification executable accepts exactly one model and one output containing
1000 finite FLOAT32 logits, with quantization `NONE`. Supported shapes are a
rank-one `(1000,)` vector or rank 2–4 with batch one, one 1000-class axis and all
other axes singleton, for example `(1,1000)`, `(1,1000,1,1)` or `(1,1,1,1000)`.
It does not assume that axis 1 always holds classes. Other class counts, integer
outputs, flattened spatial maps and batched outputs fail explicitly.

The output requires a positive physical allocation size. X5 reads class strides
from aligned dimensions; S reads byte strides. Padding is skipped, and the last
class must fit in the allocation before any read. Stable softmax uses a double
accumulator; Top-5 returns probability descending, then class ID ascending for
exact equal probabilities. This defines ties and can differ from the historical
unspecified ordering. Labels preserve the source's 1000-entry ImageNet order in
`common/imagenet_labels.h`; classification math lives in `common/classification.h`.
No manual dequantization is performed.

Model and output owners release acquired resources on early returns and C++
exceptions. Allocation and output-cache invalidation failures stop decoding.
Malformed descriptors/nonfinite logits produce a nonzero exit with a diagnostic.

Classification preprocessing uses **letterbox with gray 127 padding** in C++;
Python classification on S and Python YOLO26 uses stretch. To use stretch in C++,
set `PREPROCESS_TYPE` to `RESIZE_TYPE` and rebuild. The program selects the task
from its executable and model metadata.

<a id="pose-segment-output-contract"></a>
## Pose and segmentation output contract

Both entries require square input dimensions divisible by 32, batch one and
unquantized FLOAT32 NHWC outputs. Roles are matched by spatial shape and channels,
not output enumeration: strides 8/16/32 each contain classes (pose 1, segment 80),
boxes (4 direct LTRB or 64 DFL) and extras (pose 51 keypoint values, segment 32 mask
coefficients). All three scales must use the same box encoding. Segment adds one
stride-4, 32-channel NHWC prototype. Missing/duplicate roles, mixed encodings,
integer/SCALE tensors, unsupported layouts and nonfinite values fail explicitly.
In particular this C++ prototype path does **not** accept Python's NCHW prototype
alternative; use the matching NHWC export or the Python entry.

X5 aligned dimensions and S byte strides determine physical reads. After cache
invalidation, valid values are copied to owned compact NHWC vectors, skipping
padding. This adds a temporary host copy of the valid outputs. Allocation and
cache errors abort; acquired output/model resources release on all exit paths.
DFL and direct-distance math now use `common/decode.h` rather than private copies.
The existing keypoint equations, NMS and rendering policies remain unchanged.

Both programs discard boxes crossing the model-input boundary. Segment uses
class-agnostic NMS and renders a **model-input-sized** three-panel image
(detections, colored mask, combined), with total width `3 * input_width`. Pose
renders on the original image using its resize/padding arithmetic. A failed image
save produces a nonzero exit.

<a id="results-interpretation"></a>
## Results and verification

Boxes, masks and keypoints are drawn into result images; classification prints classes and probabilities. Check the process return code and inspect the output image. Interpret IDs in the source's built-in class order. Dataset scoring is described in the evaluator guide.

Bounded detection benchmark:

```bash
/tmp/ultralytics-cpp-detect/ultralytics_yolo_detect \
  samples/vision/ultralytics_yolo/model/yolo26n_detect_bayese_640x640_nv12.bin \
  samples/vision/ultralytics_yolo/test_data/bus.jpg /tmp/cpp-result.jpg \
  --benchmark --warmup 20 --runs 200 --rounds 3 \
  --pipeline-streams 1 --opencv-threads all \
  --score 0.25 --nms 0.7 --no-save --json /tmp/yolo-e2e-cpp.json
```
Timing starts with an in-memory BGR image and ends with restored detections: resize/letterbox, NV12 conversion, copy/cache operations, BPU and decode/NMS are included; model load, file I/O, drawing and saving are excluded. Set `--pipeline-streams 2` for two complete pipelines. Throughput is total completed frames over shared wall time; latency is per request. OpenCV thread count is independent of stream count, with no CPU-affinity restriction. Keep C++/Python, runtime-only/end-to-end and single/multistream measurements separate.

The common input owner validates metadata before allocating. Packed X5 NV12
accepts RGB-shaped NCHW/NHWC descriptors only when the physical shape is compact;
padded packed storage is explicitly rejected. Split Y/UV accepts exact batch-one
geometry and unquantized byte planes. Dynamic `-1` strides/capacity are resolved
from the actual row pitch; zero/overlapping strides, insufficient allocation and
overflow are rejected. `upload_planes` takes exact-length compact Y and interleaved
UV buffers, writes the allocated row pitch and cleans both caches. Existing I420
`upload` delegates to that path. Uploads before successful allocation or with a
different plan fail; partial allocation releases all acquired buffers.

Synchronous inference requires a non-null task handle, releases returned tasks
on creation/submission/wait failures and preserves the first error code. UCP
submission selects `HB_UCP_BPU_CORE_ANY`. Build on the target with matching DNN/UCP
SDK headers and libraries. If model binding rejects an artifact, check its target,
task, layout, dtype and output head against the selected program.

The shared `OutputTensorOwner` retains an acquired buffer for cleanup when an SDK allocation returns an error with a non-null address, and rejects success with a null address.

The shared `common/dnn_io.h` also exposes `infer_tensors_sync` for callers such as
Paraformer with more than two already validated input tensors. It performs only
synchronous SDK submission/wait/release; the caller owns and validates the full
array against model metadata. The image-facing `infer_sync` retains its one/two
input restriction, so this does not expand YOLO image input protocols.
