English | [简体中文](./README_cn.md)

# Ultralytics YOLO C++ Sample

This directory keeps the C++ reference runtime for Ultralytics YOLO on RDK X5.

## Overview

The recommended general-purpose runtime path is `runtime/python`. The
`detect` executable is an on-board RDK X5 implementation for the reviewed
YOLO26 direct-LTRB output contract. It validates the six FLOAT32 NHWC heads at
runtime and must not be used for DFL-head models without adapting the decoder.

## Directory Structure

```bash
.
|-- classify/   # Classification reference
|-- detect/     # Detection reference
|-- pose/       # Pose reference
`-- segment/    # Segmentation reference
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

## Notes

- `detect` currently implements YOLO26 direct-LTRB detection on X5.
- `classify`, `pose`, and `segment` remain reference implementations.
- Keep Python and C++ benchmark records separate because their host-side
  preprocessing, runtime wrappers, and postprocessing implementations differ.
