# Ultralytics YOLO C++ runtime

[简体中文](README_cn.md) · [Python](../python/README.md)

<a id="supported-boards"></a>
## Implementation scope and board status

The current sources implement packed NV12 `.bin` input for X5 and split Y/UV `.hbm` input for S100/S100P/S600, selecting protocols from model metadata. C++ is no longer X5-only code. Implementation and board verification are separate: no board build, run or benchmark was performed in this round. Treat this implementation scope as supported-not-run; Python evidence cannot certify C++.

| Program | Implemented protocol | Limits |
|---|---|---|
| detect | YOLO26 four-channel LTRB; YOLO11-family 64-channel DFL | Explicit head selection and benchmark |
| pose | DFL/LTRB boxes and corresponding keypoint encoding | Functional reference, no benchmark CLI |
| segment | DFL/LTRB boxes and mask coefficients/prototypes | Functional reference, no benchmark CLI |
| classify | Classification logits | Prints Top-5, no result image |

No C++ OBB entry is supplied. This does not claim the Python S YOLOv10 NMS-free semantics are covered by C++. Custom class/model layouts require checking hardcoded labels and dimensions in the sources; a matching extension is insufficient. Use the Python entry above for the complete task interface.

<a id="dependencies"></a>
## Dependencies

Board builds require CMake ≥3.10, a C++11 compiler, OpenCV development files and the matching board DNN SDK. CMake first checks `/usr/include/dnn/hb_dnn.h` and `/usr/lib`; otherwise it uses `/usr/include/hobot`, `/usr/include/hobot/dnn`, `/usr/hobot/include`, `/usr/hobot/lib` and links `hbucp`. These come from the matching board image/SDK, not the host OE model compiler. The sources do not certify every SDK version.

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

Detect/pose/segment take positional model, image and result paths; classify takes model and image only. Pose/segment/classify do not implement Python-style `--platform`, `--model-path` or general `--help`; these strings would be treated as positional paths. Legacy built-in paths depend on historical directories, so pass paths explicitly.

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

These are standalone reference executables, not a stable library API identical to Python. `DetectRuntime` in `detect/main.cc` owns model/tensors; `common/dnn_io` binds input, copies NV12 and cleans the input cache. After execution, output cache handling precedes decode/NMS and restoration to original-image coordinates. Tensors/models must not be released before requests complete.

Each concurrent stream owns its runtime context and tensors; do not reuse buffers across unfinished requests. Geometry, head probing/decode and benchmark bookkeeping live in `common/`; pose/segment/classify retain their own main programs and resource flow. Integrations must check SDK return codes, strides and valid shapes, not merely copy the inference call.

<a id="results-interpretation"></a>
## Results and verification

Boxes, masks and keypoints are drawn into result images; classification prints classes and probabilities. Check successful process exit plus the actual new output and its content; an old file at the same path is not evidence of success. Interpret IDs in the source's built-in class order. One image does not establish dataset accuracy.

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

These four pure helper tests need no board SDK and validate decode/head probing, NV12 geometry and benchmark bookkeeping only:

```bash
cmake -S samples/vision/ultralytics_yolo/runtime/cpp/test -B /tmp/ultralytics-cpp-host
cmake --build /tmp/ultralytics-cpp-host
ctest --test-dir /tmp/ultralytics-cpp-host --output-on-failure
```

Missing OpenCV development files or DNN/UCP headers/libraries cause build failures; check the target SDK rather than copying another platform's libraries. For protocol rejection inspect model target, task, layout, dtype and head. Passing host helper tests does not prove all four board executables build/run or establish new performance measurements.
