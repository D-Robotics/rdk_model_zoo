English | [简体中文](./README_cn.md)

# Ultralytics YOLO C++ Sample

This directory keeps the C++ reference runtime for Ultralytics YOLO on RDK X5
and RDK S100/S100P/S600.

## Overview

The recommended general-purpose runtime path is `runtime/python`. The C++
binaries are on-board reference implementations for both supported BPU input
protocols, probed from the model at load time: packed NV12 (X5 `.bin`, one
tensor) and split Y/UV NV12 (S-series `.hbm`, two UINT8 tensors).

Head contracts are likewise selected per model:

- `detect` auto-detects YOLO26 direct-LTRB heads (4-channel box maps) and
  YOLO11-family DFL heads (YOLOv5u/v8/v9/yolo11/yolo12/yolov13, 64-channel box
  maps); use `--head auto|dfl|ltrb` to override the auto-detection.
- `pose` and `segment` dispatch on the same box-map contract and apply the
  matching keypoint/mask encoding (YOLO26 regresses keypoints directly from
  the grid centre; YOLO11-family uses the doubled-offset encoding).
- `classify` has no head contract.

## Directory Structure

```bash
.
|-- classify/   # Classification reference
|-- common/     # Shared decode / NV12 / benchmark helpers
|-- detect/     # Detection reference
|-- pose/       # Pose reference
|-- segment/    # Segmentation reference
`-- test/       # Host-side unit tests for common/ (no board needed)
```

Each subdirectory contains its own `main.cc` and `CMakeLists.txt`.

## Build

Use the task subdirectory that you want to inspect or build.

Detection example:

```bash
cd runtime/cpp/detect
mkdir -p build
cd build
cmake ..
make
```

Classification example:

```bash
cd runtime/cpp/classify
mkdir -p build
cd build
cmake ..
make
```

## Run

Functional validation:

```bash
./build/ultralytics_yolo_detect \
  yolo26n_detect_bayese_640x640_nv12.bin \
  bus.jpg cpp_result.jpg
```

Bounded end-to-end benchmark:

```bash
./build/ultralytics_yolo_detect \
  yolo26n_detect_bayese_640x640_nv12.bin \
  bus.jpg cpp_result.jpg \
  --benchmark --warmup 20 --runs 200 --rounds 3 \
  --pipeline-streams 1 --opencv-threads all \
  --score 0.25 --nms 0.7 --no-save \
  --json e2e_cpp.json
```

Two concurrent complete pipelines:

```bash
./build/ultralytics_yolo_detect \
  yolo26n_detect_bayese_640x640_nv12.bin \
  bus.jpg cpp_result.jpg \
  --benchmark --warmup 20 --runs 200 --rounds 3 \
  --pipeline-streams 2 --opencv-threads all \
  --score 0.25 --nms 0.7 --no-save \
  --json e2e_cpp_streams2.json
```

The end-to-end scope starts with an in-memory BGR image and ends when restored
detections are ready. It includes resize/letterbox, BGR-to-NV12 conversion,
input copy and cache clean, BPU execution, output cache invalidation,
direct-LTRB decoding, class-wise NMS, and coordinate restoration. Model load,
image file I/O, drawing, and result saving are excluded.

This end-to-end command does not restrict CPU affinity and uses all online CPUs
for OpenCV preprocessing. `--pipeline-streams` controls complete preprocessing,
Runtime submission, and postprocessing pipelines; every stream owns independent
input/output tensors and a Runtime context. Multi-stream throughput is total
completed frames divided by common wall time, while latency remains a
per-request measurement. The process-wide OpenCV CPU thread setting is
independent of both the pipeline count and Runtime-only submission threads.

## Host Tests

The pure helpers in `common/` (decode math, NV12 plane geometry, benchmark
JSON) build and run on any host with a C++11 compiler:

```bash
cd runtime/cpp/test
cmake -S . -B build-host
cmake --build build-host
ctest --test-dir build-host --output-on-failure
```

## Notes

- `detect` implements the bounded end-to-end benchmark with single- and
  dual-stream pipelines for both head contracts on both input protocols.
- `classify`, `pose`, and `segment` are functional references; they accept
  both input protocols and both head contracts but do not embed the
  benchmark harness.
- S-series binaries link the board's `dnn` stack; build them on the target
  board the same way as on X5.
- Keep Python and C++ benchmark records separate because their host-side
  preprocessing, runtime wrappers, and postprocessing implementations differ.
